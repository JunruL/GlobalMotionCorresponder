python generate_dino.py --data_dir datasets/gmc_synthetic_scenes/ball/start/train
python generate_dino.py --data_dir datasets/gmc_synthetic_scenes/ball/end/train

python train_gs.py \
    -s datasets/gmc_synthetic_scenes/ball/start \
    --semantic_feature_dir datasets/gmc_synthetic_scenes/ball/start/train/dino_features \
    --rp_file datasets/gmc_synthetic_scenes/ball/start/train/dino_features/features_rp_dim16.joblib \
    --pca_file datasets/gmc_synthetic_scenes/ball/start/train/dino_features/features_pca_dim4.joblib \
    -m output/single_fgs/gmc_synthetic_scenes/ball/dinov1s8_rp16_pca4_f1\
    --white_background

python train_gs.py \
    -s datasets/gmc_synthetic_scenes/ball/end \
    --semantic_feature_dir datasets/gmc_synthetic_scenes/ball/end/train/dino_features \
    --rp_file datasets/gmc_synthetic_scenes/ball/start/train/dino_features/features_rp_dim16.joblib \
    --pca_file datasets/gmc_synthetic_scenes/ball/start/train/dino_features/features_pca_dim4.joblib \
    -m output/single_fgs/gmc_synthetic_scenes/ball/dinov1s8_rp16_pca4_f2 \
    --white_background

python train_gmc.py configs/gmc_synthetic_scenes/ball.yaml

python vis_gmc.py configs/gmc_synthetic_scenes/ball.yaml

python eval_gmc.py configs/gmc_synthetic_scenes/ball.yaml