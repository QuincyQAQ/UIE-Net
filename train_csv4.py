import os  
import warnings
import csv
import torch
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from config import Config
from data import get_training_data, get_test_data
from models import *
from utils import seed_everything, save_checkpoint
from torchvision.utils import save_image

warnings.filterwarnings('ignore')

opt = Config('config.yml')
seed_everything(opt.OPTIM.SEED)

def get_next_exp_folder(base_dir="runs/exp"):
    exp_num = 1
    while os.path.exists(f"{base_dir}{exp_num}"):
        exp_num += 1
    exp_path = f"{base_dir}{exp_num}"
    print(f"Creating experiment folder at: {exp_path}")  # Debugging output
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def train():
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        exp_folder = get_next_exp_folder()
    else:
        exp_folder = None
    metrics_file = os.path.join(exp_folder, "metrics.csv") if exp_folder else "metrics.csv"

    if accelerator.is_local_main_process:
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "PSNR", "SSIM", "LPIPS", "MAE", "best_PSNR"])

    kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator(kwargs_handlers=kwargs)
    config = {
        "dataset": opt.TRAINING.TRAIN_DIR,
        "model": opt.MODEL.SESSION
    }
    accelerator.init_trackers("uw", config=config)

    criterion_psnr = torch.nn.MSELoss()
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
    val_dataset = get_test_data(val_dir, {'w': opt.TESTING.PS_W, 'h': opt.TESTING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    model = Model()

    optimizer = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL,
                            betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    start_epoch = 1
    best_psnr = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0

        for i, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            tar = data[1]
            inp = data[0].contiguous()

            optimizer.zero_grad()
            res = model(inp)

            loss_psnr = sum([criterion_psnr(res[j], tar) for j in range(len(res))])
            loss_ssim = sum([(1 - structural_similarity_index_measure(res[j], tar, data_range=1)) for j in range(len(res))])

            train_loss = loss_psnr + 0.4 * loss_ssim

            accelerator.backward(train_loss)
            optimizer.step()

        scheduler.step()

        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            with torch.no_grad():
                psnr = 0
                ssim = 0
                lpips = 0
                mae = 0
                for _, test_data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                    tar = test_data[1]
                    inp = test_data[0].contiguous()

                    res = model(inp)[0].clamp(0, 1)

                    # Calculate metrics
                    psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                    ssim += structural_similarity_index_measure(res, tar, data_range=1)
                    mae += mean_absolute_error(torch.mul(res, 255), torch.mul(tar, 255))
                    lpips += criterion_lpips(res, tar).item()

                psnr /= len(testloader)
                ssim /= len(testloader)
                mae /= len(testloader)
                lpips /= len(testloader)

                if psnr > best_psnr:
                    best_psnr = psnr
                    if accelerator.is_local_main_process:
                        save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, epoch, exp_folder, opt.MODEL.SESSION)

                if accelerator.is_local_main_process:
                    with open(metrics_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, train_loss.item(), psnr.item(), ssim.item(), lpips, mae.item(), best_psnr.item()])

                    print(f"Epoch: {epoch}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}, MAE: {mae:.4f}, Best PSNR: {best_psnr:.4f}")

    accelerator.end_training()

if __name__ == '__main__':
    train()
