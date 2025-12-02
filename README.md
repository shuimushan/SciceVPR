# SciceVPR

This is the official code for the paper **"SciceVPR: Stable Cross-Image Correlation Enhanced Model for Visual Place Recognition"**, which is going to be published in Neurocomputing.

## Dataset

For downloading and organizing the datasets, please refer to: [CricaVPR](https://github.com/Lu-Feng/CricaVPR).

## Train

### Super-CricaVPR

python train_super_cricavpr.py \
    --eval_datasets_folder=/path/to/your/datasets_vg/datasets \
    --eval_dataset_name=pitts30k \
    --epochs_num=10 \
    --backbone_arch='dinov2_vitb14' \
    --layer1=11 \
    --out_indices 8 9 10 11 \
    --backbone_out_dim=3072 \
    --mix_in_dim=768 \
    --token_num 2 \
    --token_ratio 1 \
    --train_batch_size=72

### SciceVPR

python train_scicevpr.py \
    --eval_datasets_folder=/path/to/your/datasets_vg/datasets \
    --eval_dataset_name=pitts30k \
    --epochs_num=1 \
    --backbone_arch='dinov2_vitb14' \
    --layer1=11 \
    --out_indices 8 9 10 11 \
    --backbone_out_dim=3072 \
    --mix_in_dim=768 \
    --token_num 2 \
    --token_ratio 1 \
    --train_batch_size=72 \
    --crica_path /path/to/your/super_cricavpr.pth

### Test

python eval.py \
    --eval_datasets_folder=/path/to/your/datasets_vg/datasets \
    --eval_dataset_name=pitts30k \
    --backbone_arch='dinov2_vitb14' \
    --out_indices 8 9 10 11 \
    --backbone_out_dim=3072 \
    --mix_in_dim=768 \
    --token_num 2 \
    --token_ratio 1 \
    --pca_dim=4096 \
    --pca_dataset_folder=pitts30k/images/train \
    --resume /path/to/your/best_scicevpr.pth

### Related Work

The code for DJIST, a sequential VPR method that adopts the multi-layer feature fusion proposed by SciceVPR, will also be released soon: [DJIST](https://github.com/shuimushan/DJIST).

### Acknowledgements

Parts of this repository are inspired by the following repositories:
[CricaVPR](https://github.com/Lu-Feng/CricaVPR)
[DINO-Mix](https://github.com/GaoShuang98/DINO-Mix)
