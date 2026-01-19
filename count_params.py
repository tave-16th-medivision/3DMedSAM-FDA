"""
Usage example:
python count_params.py
"""

import torch
import torch.nn as nn
from functools import partial

# 모델 경로 수정 필요 시 변경
from modeling.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
from modeling.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer

def count_parameters():
    print("Building models and counting parameters (Target: 1 pt/volume)...\n")

    # 1. Image Encoder (변동 없음)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice=16
    )

    # Gradient 설정 (변동 없음)
    for p in img_encoder.parameters(): p.requires_grad = False
    img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.slice_embed.parameters(): p.requires_grad = True
    for i in img_encoder.blocks:
        for p in i.norm1.parameters(): p.requires_grad = True
        for p in i.adapter.parameters(): p.requires_grad = True
        for p in i.norm2.parameters(): p.requires_grad = True
        if hasattr(i.attn, 'rel_pos_d') and isinstance(i.attn.rel_pos_d, nn.Parameter):
            i.attn.rel_pos_d.requires_grad = True
        else:
             i.attn.rel_pos_d = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)
    for i in img_encoder.neck_3d:
        for p in i.parameters(): p.requires_grad = True

    # ==========================================================
    # [수정됨] Prompt Encoder List: range(4) -> range(1)
    # 1 pt/volume 설정에 맞춤
    # ==========================================================
    prompt_encoder_list = []
    # 원래 코드: for i in range(4):
    for i in range(1): 
        prompt_encoder = PromptEncoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8
            )
        )
        prompt_encoder_list.append(prompt_encoder)

    # 3. Mask Decoder (변동 없음)
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)

    # 4. 파라미터 계산
    def get_params(model_iter):
        tuned = 0
        total = 0
        for p in model_iter:
            total += p.numel()
            if p.requires_grad:
                tuned += p.numel()
        return tuned, total

    img_tuned, img_total = get_params(img_encoder.parameters())
    prompt_tuned, prompt_total = get_params((p for module in prompt_encoder_list for p in module.parameters()))
    dec_tuned, dec_total = get_params(mask_decoder.parameters())

    all_tuned = img_tuned + prompt_tuned + dec_tuned
    all_total = img_total + prompt_total + dec_total

    # 5. 출력
    print("=" * 65)
    print(f"{'Component':<20} | {'Tuned (M)':<12} | {'Total (M)':<12} | {'Ratio (%)':<10}")
    print("-" * 65)
    
    print(f"{'Image Encoder':<20} | {img_tuned/1e6:<12.2f} | {img_total/1e6:<12.2f} | {img_tuned/img_total*100:<10.2f}")
    # (x1) 로 표시 변경
    print(f"{'Prompt Encoder (x1)':<20} | {prompt_tuned/1e6:<12.2f} | {prompt_total/1e6:<12.2f} | {prompt_tuned/prompt_total*100:<10.2f}")
    print(f"{'Mask Decoder':<20} | {dec_tuned/1e6:<12.2f} | {dec_total/1e6:<12.2f} | {dec_tuned/dec_total*100:<10.2f}")
    
    print("-" * 65)
    print(f"{'SUM TOTAL':<20} | {all_tuned/1e6:<12.2f} | {all_total/1e6:<12.2f} | {all_tuned/all_total*100:<10.2f}")
    print("=" * 65)

if __name__ == "__main__":
    count_parameters()