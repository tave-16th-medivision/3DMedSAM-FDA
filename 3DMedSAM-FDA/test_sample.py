"""
Usage example:
python test_sample.py \
  --img_path /workspace/dataset/Task10_Colon/imagesTr/colon_001.nii.gz \
  --mask_path /workspace/dataset/Task10_Colon/labelsTr/colon_001.nii.gz \
  --snapshot_path /workspace/weights/ \
  --data colon \
  --case_id 001 \
  --output_dir /workspace/samples/colon 
"""


import argparse
import numpy as np
import logging
import os
import torch
import torch.nn.functional as F
import nibabel as nib
from functools import partial
from monai.losses import DiceLoss
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, ScaleIntensityRangePercentiles
from monai.data import MetaTensor

# 모델 관련 임포트 (기존 test.py와 동일)
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
from monai.transforms import ScaleIntensityRange


def load_single_case(img_path, mask_path):
    """단일 케이스를 로드하고 전처리를 수행합니다."""
    # image_only=True로 설정하여 MetaTensor 반환
    loader = LoadImage(ensure_channel_first=True, image_only=True)
    
    img = loader(img_path)
    mask = loader(mask_path)
    
    affine = img.affine
    
    # 2. 전처리
    orient = Orientation(axcodes="RAS")
    img = orient(img)
    mask = orient(mask)
    
    # Array/Tensor용 변환 함수를 사용합니다.
    scaler = ScaleIntensityRange(
        a_min=-175, 
        a_max=250, 
        b_min=0.0, 
        b_max=1.0, 
        clip=True
    )
    
    img = scaler(img)
    
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    
    # Spacing 추출
    if hasattr(img, "pixdim"):
        spacing = img.pixdim
        if hasattr(spacing, "cpu"):
            spacing = spacing.cpu().numpy()
        else:
            spacing = np.array(spacing)

        if len(spacing) > 3:
            spacing = spacing[1:4]
    else:
        spacing = np.array(img.meta["pixdim"][1:4])
    
    return img, mask, spacing, affine

def save_nifti(data, affine, path):
    """Numpy 배열을 NIfTI 파일로 저장"""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # boolean 마스크라면 uint8로 변환
    if data.dtype == bool:
        data = data.astype(np.uint8)
        
    # 차원 정리 (C, D, H, W) -> (D, H, W)
    if data.ndim == 4:
        data = data[0]
    elif data.ndim == 5:
        data = data[0, 0]
        
    ni_img = nib.Nifti1Image(data, affine)
    nib.save(ni_img, path)
    print(f"Saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    # 필수 경로 설정
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to ground truth mask")
    parser.add_argument("--output_dir", type=str, default="samples/colon", help="Directory to save results")
    parser.add_argument("--case_id", type=str, default="001", help="Case ID for logging")
    
    # 기존 모델 설정
    parser.add_argument("--snapshot_path", default="", type=str, required=True)
    parser.add_argument("--data", default="colon", type=str)
    parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_prompts", default=1, type=int)
    parser.add_argument("--checkpoint", default="best", type=str) # 보통 best 사용
    parser.add_argument("-tolerance", default=5, type=int)
    
    args = parser.parse_args()
    
    # 경로 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
        
    # Snapshot 경로 보정
    if args.data not in args.snapshot_path:
         args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    device = args.device

    # Crop Size 설정
    if args.rand_crop_size == 0:
        args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)

    # 로거 설정
    setup_logger(logger_name="test_sample", root=args.output_dir, screen=True, tofile=False)
    logger = logging.getLogger(f"test_sample")
    
    # === 모델 초기화 (기존 코드와 동일) ===
    img_encoder = ImageEncoderViT_3d(
        depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12, patch_size=16, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11], window_size=14, cubic_window_size=8,
        out_chans=256, num_slice=16
    )
    # Checkpoint 로드
    ckpt_path = os.path.join(args.snapshot_path, file)
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    img_encoder.load_state_dict(checkpoint["encoder_dict"], strict=True)
    img_encoder.to(device)

    prompt_encoder_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8))
        prompt_encoder.load_state_dict(checkpoint["feature_dict"][i], strict=True)
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)

    mask_decoder = VIT_MLAHead(img_size=96).to(device)
    mask_decoder.load_state_dict(checkpoint["decoder_dict"], strict=True)
    mask_decoder.to(device)

    dice_loss_func = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
    
    img_encoder.eval()
    for p in prompt_encoder_list: p.eval()
    mask_decoder.eval()

    # === 데이터 로드 (단일 케이스) ===
    logger.info(f"Processing Case {args.case_id}...")
    img, seg, spacing, affine = load_single_case(args.img_path, args.mask_path)
    
    patch_size = args.rand_crop_size[0]

    # === Inference 함수 수정 ===
    def model_predict(img, prompt, img_encoder, prompt_encoder, mask_decoder):
        # 1. 이미지 크기 조정
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        
        # 2. 배치 차원 변경 (Batch, Channel, D, H, W) -> (D, Channel, H, W) 처럼 변경됨
        input_batch = out[0].transpose(0, 1)
        
        # [수정된 부분] 채널이 1개라면 3개로 복사 (SAM Encoder 입력 조건 맞추기)
        if input_batch.shape[1] == 1:
            input_batch = input_batch.repeat(1, 3, 1, 1)
            
        # 3. 인코더 통과
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        
        points_torch = prompt.transpose(0, 1)
        new_feature = []
        for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder)):
            if i == 3:
                new_feature.append(
                    feature_decoder(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size])
                )
            else:
                new_feature.append(feature.to(device))
        
        # 4. 이미지 리사이즈 및 마스크 디코더 입력 준비
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size, mode="trilinear")
        new_feature.append(img_resize)
        
        masks = mask_decoder(new_feature, 2, patch_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks

    # === 추론 실행 ===
    with torch.no_grad():
        seg = seg.float()
        prompt = F.interpolate(seg, img.shape[2:], mode="nearest")[0] # (1, D, H, W) -> (C, D, H, W) interpolation
        
        seg = seg.to(device)
        img = img.to(device)
        seg_pred = torch.zeros_like(prompt).to(device)

        # Prompt 생성 로직 (Random Point Selection)
        l = len(torch.where(prompt == 1)[0])
        # Random seed 고정 (재현성 위해)
        np.random.seed(42) 
        if l > 0:
            sample_indices = np.random.choice(np.arange(l), args.num_prompts, replace=True)
            
            # 좌표 추출 (Interpolated space)
            # prompt shape: (1, D, H, W) -> indices are (0, z, y, x) or similar depending on implementation
            # torch.where returns tuple of tensors per dimension
            # dim 0 is channel (always 0), dim 1 is D(x?), dim 2 is H(y?), dim 3 is W(z?)
            # 기존 코드: x=dim1, z=dim2, y=dim3 로 매핑하여 사용중 (Data format에 따라 다름)
            
            p_coords = torch.where(prompt == 1)
            x = p_coords[1][sample_indices].unsqueeze(1)
            y = p_coords[3][sample_indices].unsqueeze(1) # 기존 코드 로직 따름 (dim 3)
            z = p_coords[2][sample_indices].unsqueeze(1) # 기존 코드 로직 따름 (dim 2)
            
            # Point Volume 생성을 위한 좌표 저장 (원본 공간 좌표)
            points_for_save = []
            for i in range(args.num_prompts):
                points_for_save.append([x[i].item(), z[i].item(), y[i].item()]) # x, z, y 순서 주의 (기존 코드 로직 기반)

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

            points = torch.cat([x-d_min, y-w_min, z-h_min], dim=1).unsqueeze(1).float()
            points_torch = points.to(device)
            
            d_min = max(0, d_min)
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            
            img_patch = img[:, :, d_min:d_max, h_min:h_max, w_min:w_max].clone()
            img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
            
            # 모델 예측
            pred = model_predict(img_patch, points_torch, img_encoder, prompt_encoder_list, mask_decoder)
            
            pred = pred[:,:, d_l:patch_size-d_r, h_l:patch_size-h_r, w_l:patch_size-w_r]
            pred = F.softmax(pred, dim=1)[:,1]
            seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred
        else:
            logger.warning("No tumor found in mask, skipping inference.")
            points_for_save = []

        # 최종 예측 마스크 생성
        final_pred = F.interpolate(seg_pred.unsqueeze(1), size=seg.shape[2:], mode="trilinear")
        masks = final_pred > 0.5
        
        # Metric 계산
        loss = 1 - dice_loss_func(masks, seg)
        dice_score = loss.item()
        
        ssd = surface_distance.compute_surface_distances(
            (seg == 1)[0, 0].cpu().numpy(),
            (masks == 1)[0, 0].cpu().numpy(),
            spacing_mm=spacing  
        )
        nsd_score = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)

        logger.info(f"Result - Dice: {dice_score:.4f}, NSD: {nsd_score:.4f}")

    # === 결과 저장 ===
    # 1. Image 저장
    save_nifti(img, affine, os.path.join(args.output_dir, f"imaging_{args.case_id}.nii.gz"))
    
    # 2. Mask 저장
    save_nifti(seg, affine, os.path.join(args.output_dir, f"mask_{args.case_id}.nii.gz"))
    
    # 3. Prediction 저장
    save_nifti(masks, affine, os.path.join(args.output_dir, f"prediction_{args.case_id}.nii.gz"))
    
    # 4. Point Prompt 저장 (3D Volume에 점 찍기)
    point_vol = np.zeros(masks.shape[2:], dtype=np.uint8) # (D, H, W)
    for pt in points_for_save:
        # pt는 [x, z, y] 순서 (위 코드 로직 따름)
        # 인덱스 범위 체크 후 마킹
        px, pz, py = int(pt[0]), int(pt[1]), int(pt[2])
        if 0 <= px < point_vol.shape[0] and 0 <= pz < point_vol.shape[1] and 0 <= py < point_vol.shape[2]:
            # 잘 보이게 하기 위해 3x3x3 영역 마킹 (선택 사항)
            point_vol[max(0, px-1):px+2, max(0, pz-1):pz+2, max(0, py-1):py+2] = 1
            
    save_nifti(point_vol, affine, os.path.join(args.output_dir, f"point_{args.case_id}.nii.gz"))

    # 5. Log 파일 저장
    log_path = os.path.join(args.output_dir, f"sample_{args.case_id}.log")
    with open(log_path, "w") as f:
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Case ID: {args.case_id}\n")
        f.write(f"Image Path: {args.img_path}\n")
        f.write(f"Mask Path: {args.mask_path}\n")
        f.write(f"Number of Prompts: {args.num_prompts}\n")
        f.write(f"Prompt Locations (Indices [D, H, W]): {points_for_save}\n")
        f.write(f"Dice Score: {dice_score:.6f}\n")
        f.write(f"NSD Score: {nsd_score:.6f}\n")
    
    print(f"로그 파일 저장 완료: {log_path}")

if __name__ == "__main__":
    main()