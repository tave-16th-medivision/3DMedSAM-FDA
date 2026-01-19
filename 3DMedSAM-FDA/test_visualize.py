"""
Usage example:
python -u test_visualize.py --data colon --snapshot_path "/workspace/weights/" --data_prefix "/workspace/dataset/Task10_Colon/" --num_prompts 1 
"""

from dataset.datasets import load_data_volume
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics

# --- Matplotlib 백엔드 설정 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
# ---------------------------

def visualize_and_save(img_3d, gt_3d, pred_3d, points_3d, prompt_z_index, case_name, save_dir):
    """
    points_3d: (Height, Width, Depth) 순서로 정렬된 좌표를 기대합니다.
    """
    # 1. 해상도 맞추기
    target_size = img_3d.shape[2:]
    if gt_3d.shape[2:] != target_size:
        gt_3d = F.interpolate(gt_3d.float(), size=target_size, mode='nearest')
    if pred_3d.shape[2:] != target_size:
        pred_3d = F.interpolate(pred_3d.float(), size=target_size, mode='nearest')

    img_np = img_3d[0, 0].detach().cpu().numpy()
    gt_np = gt_3d[0, 0].detach().cpu().numpy().astype(np.uint8)
    pred_np = (pred_3d[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)

    # 3. 최적 슬라이드 찾기
    gt_sum = np.sum(gt_np, axis=(1, 2))
    if gt_sum.max() > 0:
        best_z = np.argmax(gt_sum)
    else:
        pred_sum = np.sum(pred_np, axis=(1, 2))
        best_z = np.argmax(pred_sum) if pred_sum.max() > 0 else prompt_z_index

    # 범위 안전 장치
    if best_z >= img_np.shape[0]: best_z = prompt_z_index
    if prompt_z_index >= img_np.shape[0]: 
        print(f"Skipping {case_name}: Index out of bounds")
        return

    # 그리기 함수
    def plot_slice(z_idx, suffix, desc):
        img_slice = img_np[z_idx, :, :]
        gt_slice = gt_np[z_idx, :, :]
        pred_slice = pred_np[z_idx, :, :]

        plt.figure(figsize=(10, 10))
        plt.imshow(img_slice, cmap='gray')

        gt_slice_cv = gt_slice * 255
        pred_slice_cv = pred_slice * 255

        contours_gt, _ = cv2.findContours(gt_slice_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_gt: 
            if len(c)>0: plt.plot(c[:,0,0], c[:,0,1], color='blue', linewidth=2, label='GT')
        
        contours_pred, _ = cv2.findContours(pred_slice_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_pred: 
            if len(c)>0: plt.plot(c[:,0,0], c[:,0,1], color='red', linewidth=2, label='Pred')

        # 포인트 표시 (Index 2가 Depth인 점들 중, 현재 슬라이드 z_idx와 일치하는 것만)
        p_np = points_3d.detach().cpu().numpy()
        p_on_slice = p_np[p_np[:, 2] == z_idx]
        if len(p_on_slice) > 0:
            # scatter(x, y) -> (Width, Height) -> (Index 1, Index 0)
            # plt.scatter(p_on_slice[:, 1], p_on_slice[:, 0], c='red', s=100, edgecolors='white', linewidth=2, label='Prompt')
            plt.scatter(p_on_slice[:, 1], p_on_slice[:, 0], c='#FF69B4', s=100, edgecolors='white', linewidth=2, label='Prompt')

        plt.axis('off')
        short_name = os.path.basename(case_name)
        plt.title(f"{short_name}\n{desc} (z={z_idx})")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: plt.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right')

        plt.savefig(os.path.join(save_dir, f"{short_name}{suffix}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    plot_slice(best_z, "_best", "Best Mask Slice")
    plot_slice(prompt_z_index, "_prompt", "Prompt Slice")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_prompts", default=1, type=int)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("--checkpoint", default="last", type=str)
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()
    
    file = "best.pth.tar" if args.checkpoint != "last" else "last.pth.tar"
    device = args.device
    
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
            
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    viz_save_dir = os.path.join(args.snapshot_path, f"test_{timestamp}")
    if not os.path.exists(viz_save_dir): os.makedirs(viz_save_dir)
    print(f"Visualization results: {viz_save_dir}")

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    
    test_data = load_data_volume(
        data=args.data, batch_size=1, path_prefix=args.data_prefix, augmentation=False, split="test",
        rand_crop_spatial_size=args.rand_crop_size, convert_to_sam=False, do_test_crop=False, deterministic=True, num_worker=0
    )
    
    img_encoder = ImageEncoderViT_3d(
        depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12, patch_size=16, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11], window_size=14, cubic_window_size=8,
        out_chans=256, num_slice = 16)
    img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)
    img_encoder.to(device)

    prompt_encoder_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8))
        prompt_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["feature_dict"][i], strict=True)
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)

    mask_decoder = VIT_MLAHead(img_size = 96).to(device)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"], strict=True)
    mask_decoder.to(device)

    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
    img_encoder.eval()
    for i in prompt_encoder_list: i.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, prompt, img_encoder, prompt_encoder, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        points_torch = prompt.transpose(0, 1)
        new_feature = []
        for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder)):
            if i == 3:
                new_feature.append(feature_decoder(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size]))
            else:
                new_feature.append(feature.to(device))
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size, mode="trilinear")
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, patch_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for idx, (img, seg, spacing) in enumerate(test_data):
            case_name = test_data.dataset.img_dict[idx]

            seg = seg.float()
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)
            seg_pred = torch.zeros_like(prompt).to(device)
            l = len(torch.where(prompt == 1)[0])
            
            sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            
            # --- [복구됨] train.py와 동일한 로직 유지 (모델 성능 보존) ---
            # 여기서 x는 Depth, z는 Height를 의미하게 됨 (변수명이 헷갈리지만 로직은 유지)
            x = torch.where(prompt == 1)[1][sample].unsqueeze(1) # Index 1 (Depth) -> x 변수
            y = torch.where(prompt == 1)[3][sample].unsqueeze(1) # Index 3 (Width) -> y 변수
            z = torch.where(prompt == 1)[2][sample].unsqueeze(1) # Index 2 (Height) -> z 변수
            # --------------------------------------------------------

            x_m = (torch.max(x) + torch.min(x)) // 2
            y_m = (torch.max(y) + torch.min(y)) // 2
            z_m = (torch.max(z) + torch.min(z)) // 2

            d_min = x_m - patch_size//2
            d_max = x_m + patch_size//2
            h_min = z_m - patch_size//2
            h_max = z_m + patch_size//2
            w_min = y_m - patch_size//2
            w_max = y_m + patch_size//2
            
            d_l = max(0, -d_min)
            d_r = max(0, d_max - prompt.shape[1])
            h_l = max(0, -h_min)
            h_r = max(0, h_max - prompt.shape[2])
            w_l = max(0, -w_min)
            w_r = max(0, w_max - prompt.shape[3])

            # 모델에는 train.py와 똑같이 (x, y, z) 순서로 들어감 (Depth, Width, Height)
            points = torch.cat([x-d_min, y-w_min, z-h_min], dim=1).unsqueeze(1).float()
            points_torch = points.to(device)
            
            d_min_idx = max(0, d_min)
            h_min_idx = max(0, h_min)
            w_min_idx = max(0, w_min)
            
            # x_m이 Depth Center이므로 Depth 슬라이싱에 사용 (올바른 로직)
            img_patch = img[:, :,  d_min_idx:d_max, h_min_idx:h_max, w_min_idx:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
            
            pred = model_predict(img_patch, points_torch, img_encoder, prompt_encoder_list, mask_decoder)
            
            pred = pred[:,:, d_l:patch_size-d_r, h_l:patch_size-h_r, w_l:patch_size-w_r]
            pred = F.softmax(pred, dim=1)[:,1]
            seg_pred[:, d_min_idx:d_max, h_min_idx:h_max, w_min_idx:w_max] += pred

            final_pred = F.interpolate(seg_pred.unsqueeze(1), size = seg.shape[2:],  mode="trilinear")
            masks = final_pred > 0.5

            # --- [수정됨] 시각화용 좌표 생성 ---
            # 시각화 함수는 (Height, Width, Depth) 순서를 표준으로 기대함
            # x변수=Depth, y변수=Width, z변수=Height 이므로 순서를 (z, y, x)로 묶어줌
            points_viz = torch.cat([z, y, x], dim=1) 
            
            # 슬라이스 인덱스는 Depth인 x_m을 사용
            visualize_and_save(img, seg, masks, points_viz, x_m.item(), case_name, viz_save_dir)
            # -------------------------------

            loss = 1 - dice_loss(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            ssd = surface_distance.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(), (masks==1)[0, 0].cpu().numpy(), spacing_mm=spacing[0].numpy())
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)
            loss_nsd.append(nsd)
            logger.info(" Case {} - Dice {:.6f} | NSD {:.6f}".format(case_name, loss.item(), nsd))
        
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))

if __name__ == "__main__":
    main()