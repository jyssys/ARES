# ARES: Auxiliary Range Expansion for Outlier Synthesis

This is the source code accompanying the paper [***ARES: Auxiliary Range Expansion for Outlier Synthesis***]


## Informations
- OS : ubuntu 20.04
- python : 3.8.10
- CUDA : 11.4
- NVIDIA Driver version : 470.82.01
- GPU : NVIDIA Geforce a10 (24GB)

## Requirements

```
pip install -r requirements.txt
```

The related version for PyTorch and related library (e.g. torchvision) is based on cu113. 
Please adjust it according to the experimental environment.

## Dataset Preparation
**Download the all image dataset from [here](https://drive.google.com/drive/folders/1DEjHpTipAGmsNOeOQ2S2fC-u_uQWVuMd?usp=sharing)**.

The dataset folder should have the following below structure:
CIFAR-10 and CIFAR-100 datasets automatically download by ARES code.
<br>
    
    └── ARES
        ├── models
        ├── ...
        └── data
            |
            ├── fractals_and_fvis
            ├── texture
            ├── place365
            ├── LSUN
            ├── LSUN_resize
            ├── iSUN
            └── test_32x32.mat


## Training
First, enter the ARES folder by running

```
cd ARES
```

then, run this command for ARES training
**this command basis on pre-train epochs (start epochs) 200 and total epochs 500**

```
python train_ARES.py --mode 'ARES' --seed 605 \
                     --start_epoch 40 --epochs 100 \
                     --m_sample_number 10000 --sample_number 1000 --sample_from 10000 \
                     --select 1 --loss_weight 0.1 \
                     --all-ops --k 4 --alpha 3 \
                     --dataset 'cifar10' \
                     --use_wandb
```

#### Training Arguments Explain

- --mode: Training mode. 'VOS' trains VOS, 'ARES' trains ARES.
- --seed: Training seed. We use three seeds: 605, 905 and 115.
- --start_epoch: Pre-train epochs for estimating virtual outliers.
- --m_sample_number: The number of mixup ID data ($\tilde{\mathbf{x}}^*$) to be collected for estimating the multivariate normal distribution in the Estimation stage.
- --sample_number: The number of ID data ($\mathbf{x}$) to be collected for estimating the multivariate normal distribution per class in VOS.
- --sample_from: The number to be randomly sampled from the estimated multivariate normal distribution in the Estimation stage.
- --select: The number of virtual outliers to create from the t-th smallest likelihood instances in the Estimation stage. This argument only use in 'VOS' mode. In 'ARES' mode, select automatically follow batch size. (In 'VOS' mode, select set 1.)
- --loss_weight: Weight for the discrimination loss in the Divergence stage.
- --alpha: The parameter of the beta distribution that determines the mixup ratio in the Expansion stage.
- --all-ops: 'store_true' value. If TRUE, use all augmentations in the Escape stage. If FALSE, use only some augmentations in the Escape stage.
- --beta: The parameter of the beta distribution that determines the mixup ratio in the Escape stage.
- --k: The number of times to perform mixup in the Escape stage.
- --dataset: Training ID data.
- --use_wandb: 'store_true' value. Determines whether to use wandb or not.

## Evaluation

- .pt result will be recorded to folder 'snapshots/baseline'.
- .csv result that recorded training duration, loss and error will be saved to folder 'snapshots/baseline'.

```
python test.py --score energy --seed 605 \
               --method_name 'cifar10_wrn_baseline_0.1_10000_200_1_10000_epoch_499_ARES_500_seed605_pre'
```

#### Evaluation Arguments Explain

- --score: How to evaluate method the OOD model. ARES evaluated by energy-based score.
- --seed: Evaluation seed. We use three seeds: 605, 905 and 115.
- --method_name: Filename of the .pt file to evaluate.

#### Pretrained models

- We provide a total of 5 pre-train models: 500 epochs for seeds 605, 905, and 115; 100 and 200 epochs for seed 605. 
- They are included in the snapshots/baseline directory of the supplementary code.