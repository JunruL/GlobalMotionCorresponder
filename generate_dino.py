import argparse
import torch
import os
import joblib

import numpy as np
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from typing import List, Tuple
from sklearn import random_projection
from sklearn.decomposition import PCA

from dino_utils.extractor import ViTExtractor

os.environ['TORCH_HOME'] = 'dino_utils/pretrained_models'


def extract_dino(data_dir, load_size: int = 224, stride: int = 4, save_dir: str = 'dump') -> List[Tuple[Image.Image, np.ndarray]]:
    """
    data_dir: directory containing input images.
    load_size: size of the smaller edge of loaded images. If None, does not resize.
    stride: stride of the model.
    :return: a list of lists containing an image and its principal components.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_paths = sorted([x for x in data_dir.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    save_dir.mkdir(exist_ok=True, parents=True)
    extractor = ViTExtractor(model_type='dino_vits8', stride=stride, device=device)

    feature_maps = []

    # extract descriptors and saliency maps for each image
    for image_path in image_paths:
        image_batch, _ = extractor.preprocess(image_path, load_size)

        descs = extractor.extract_descriptors(image_batch.to(device))
        curr_num_patches = extractor.num_patches

        descs = descs.reshape(curr_num_patches[0], curr_num_patches[1], -1)
        descs = descs.half()

        output_path = os.path.join(save_dir, image_path.name[:-4] + ".pth")
        torch.save(descs, output_path)

        feature_maps.append(descs.permute(2, 0, 1))
    
    aggregated_features = torch.stack(feature_maps, dim=0)

    return aggregated_features


def load_masks(directory):
    "Load the masks from the png files"
    files = sorted(os.listdir(directory))
    masks = []
    
    for filename in files:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            image = Image.open(path)
            image = np.array(image.convert("RGBA"))
            masks.append(torch.tensor(image[:, :, -1], dtype=torch.float32))

    aggregated_masks = torch.cat([mask[None, :, :] for mask in masks], dim=0)
    
    return aggregated_masks
    
    
def save_pca_and_rp(feature_maps, data_dir, feat_dir, pca_dim=4, rp_dim=16):
    feat_dim = feature_maps.shape[1]
    
    masks = load_masks(data_dir)
    masks = F.interpolate(masks.unsqueeze(1), size=(feature_maps.shape[2], feature_maps.shape[3]), mode='bilinear', align_corners=True).squeeze(1)
    masks = masks >= 0.5
    masks = masks.to(feature_maps.device)

    feature_maps = feature_maps.permute(1, 0, 2, 3)[:, masks].permute(1, 0).reshape(-1, feat_dim).cpu().numpy()

    pca = PCA(n_components=pca_dim, random_state=0).fit(feature_maps)
    pca_fname = os.path.join(feat_dir, f"features_pca_dim{pca_dim}.joblib")
    joblib.dump(pca, pca_fname)
    
    rp = random_projection.GaussianRandomProjection(n_components=rp_dim).fit(feature_maps)
    rp_fname = os.path.join(feat_dir, f"features_rp_dim{rp_dim}.joblib")
    joblib.dump(rp, rp_fname)

    return pca, rp
    
def vis_pca_features(feature_maps, masks, pca, vis_dir):
    masks = load_masks(data_dir)

    for i in range(feature_maps.shape[0]):
        mask = masks[i].bool()
        feature_map = feature_maps[i]

        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(mask.shape[0], mask.shape[1]), mode='bilinear', align_corners=True).squeeze(0)

        feat_dim, h, w = feature_map.shape

        feature_map_ = feature_map.permute(1, 2, 0).reshape(-1, feat_dim).cpu().numpy()
        pca_feat = pca.transform(feature_map_)
        pca_feat = pca_feat.reshape(h, w, -1)

        vis_feat = pca_feat[..., -3:]

        q1, q99 = np.percentile(vis_feat.reshape(-1, 3), [1, 99])
        vis_feat = (vis_feat - q1) / (q99 - q1)
        vis_feat = np.clip(vis_feat, 0.0, 1.0)

        vis_feat = vis_feat * mask[..., None].cpu().numpy()

        vis_feat = (vis_feat * 255).astype(np.uint8)
        img = Image.fromarray(vis_feat)
        img.save(os.path.join(vis_dir, f"{i:05d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--data_dir', type=str, required=True, help='The root dir of images.')
    parser.add_argument('--load_size', default=800, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=8, type=int, help="stride of first convolution layer. small stride -> higher resolution.")
    parser.add_argument('--pca_dim', default=4, type=int, help="dimension of the PCA.")
    parser.add_argument('--rp_dim', default=16, type=int, help="dimension of the random projection.")

    args = parser.parse_args()

    with torch.no_grad():
        data_dir = Path(args.data_dir)
        feat_dir = Path(f'{args.data_dir}/dino_features')

        feature_maps = extract_dino(data_dir, args.load_size, args.stride, feat_dir)
        pca, rp = save_pca_and_rp(feature_maps, data_dir, feat_dir, pca_dim=args.pca_dim, rp_dim=args.rp_dim)

        vis_dir = f"{feat_dir}_vis_pcadim{args.pca_dim}"
        os.makedirs(vis_dir, exist_ok=True)

        vis_pca_features(feature_maps, data_dir, pca, vis_dir)