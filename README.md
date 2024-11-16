### 📋 [UIE-UnFold: Deep Unfolding Network with Color Priors and Vision Transformer for Underwater Image Enhancement](https://ieeexplore.ieee.org/document/10722842)

<div>
<span class="author-block">
  Yingtie Lei<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    Jia Yu<sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    Yihang Dong
  </span>,
  <span class="author-block">
    Changwei Gong
  </span>,
  <span class="author-block">
    Ziyang Zhou
  </span>,
  <span class="author-block">
  Chi-Man Pun<sup>📮</sup>
</span>
  (👨‍💻‍ Equal contributions, 📮 Corresponding Author)
  </div>
<b>University of Macau, Huizhou University, SIAT CAS</b>

In ***Computational Imaging, Vision, and Language (CIVIL) @ IEEE International Conference on Data Science and Advanced Analytics 2024 (CIVIL @ DSAA 2024)***

## ⚙️ Usage
### Training
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TRAINING in config.yml

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties on the usage of accelerate, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

### Inference
Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in traning.yml
```
python test.py
```

# 💗 Acknowledgements
This work was supported by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3.

### 🛎 Citation
If you find our work helpful for your research, please cite:
```bib
```
