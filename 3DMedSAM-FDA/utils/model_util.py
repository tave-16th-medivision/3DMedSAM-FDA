def get_model(args):
    """
    모델 선택 함수 (Factory 패턴)
    - args.method 문자열에 따라 서로 다른 3D segmentation backbone을 생성하여 반환.
    - 공통 입력: 1채널(CT/MRI 볼륨), 출력: args.num_classes 채널 segmentation logit.

    Args:
        args: 실험 설정을 담는 인자 집합
            args.method       : 선택할 세그멘테이션 모델 이름
            args.rand_crop_size : 입력 볼륨의 크기 (예: [128,128,128])
            args.num_classes  : segmentation class 수

    Returns:
        seg_net (nn.Module): 선택된 segmentation model (3D)
    """
    
    # -------------------------------------------------------
    # 1) SwinUNETR (MONAI)
    #    Swin Transformer 기반 encoder + 3D UNet decoder
    # -------------------------------------------------------
    if args.method == "swin_unetr":
        from monai.networks.nets import SwinUNETR

        seg_net = SwinUNETR(
            img_size=args.rand_crop_size,    # 입력 볼륨 크기
            in_channels=1,                   # CT/MRI: 1채널 입력
            out_channels=args.num_classes,   # segmentation class 수
            feature_size=48,                 # 기본 피처 크기 (swin 축소단계에 영향)
            use_checkpoint=True,             # 메모리 절약을 위한 gradient checkpoint
        )

    # -------------------------------------------------------
    # 2) UXNET (3D Conv + Transformer Hybrid)
    #    최근 3D segmentation 성능 좋은 모델 중 하나
    # -------------------------------------------------------
    elif args.method == "3d_uxnet":
        from modeling.uxnet import UXNET

        seg_net = UXNET(
            in_chans=1,                     # 입력 채널
            out_chans=args.num_classes,     # segmentation 클래스 수
            depths=[2, 2, 2, 2],            # hierarchical depth (stage별 transformer 깊이)
            feat_size=[48, 96, 192, 384],   # stage별 channel 수
            drop_path_rate=0,               # stochastic depth 비활성화
            layer_scale_init_value=1e-6,    # LayerScale 초기값
            spatial_dims=3,                 # 3D 모델
        )

    # -------------------------------------------------------
    # 3) UNETR++ (Hybrid 3D Vision Transformer)
    #    UNETR의 개선 버전: decoder path 강화
    # -------------------------------------------------------
    elif args.method == "unetr++":
        from modeling.unetr_pp.unetr_pp import UNETR_PP

        # if args.data == "msd":
        #     patch_size = [2, 4, 4]
        # else:
        #     patch_size = [4, 4, 4]

        # MSD dataset일 때 patch_size 조절 가능 (주석 처리됨)
        # patch_size = [2,4,4] 로 고정

        seg_net = UNETR_PP(
            in_channels=1,
            out_channels=args.num_classes,
            img_size=args.rand_crop_size,     # 입력 크기
            patch_size=[2, 4, 4],             # Transformer patch 분해 크기 (3D)
            feature_size=16,                  # encoder base feature
            num_heads=4,                      # multi-head attention heads
            depths=[3, 3, 3, 3],              # Transformer layer depth per stage
            dims=[32, 64, 128, 256],          # stage별 embedding dimension
            do_ds=False,                      # deep supervision 사용 X
        )


    # -------------------------------------------------------
    # 4) 원본 MONAI UNETR
    #    ViT encoder + conv decoder
    # -------------------------------------------------------
    elif args.method == "unetr":
        from modeling.unetr import UNETR

        seg_net = UNETR(
            in_channels=1,
            out_channels=args.num_classes,
            img_size=args.rand_crop_size,
            patch_size=16,                 # ViT patch 크기 (isotropic)
            feature_size=16,               # base feature size
            hidden_size=768,               # ViT embedding dimension
            mlp_dim=3072,                  # MLP dimension
            num_heads=12,                  # Transformer heads
            pos_embed="perceptron",        # positional encoding 방식
            norm_name="instance",          # normalization: InstanceNorm
            res_block=True,                # decoder residual block 사용
            dropout_rate=0.0,              # dropout 없음
        )

    # -------------------------------------------------------
    # 5) nnFormer
    #    3D segmentation을 위해 설계된 Transformer 기반 모델
    # -------------------------------------------------------
    elif args.method == "nnformer":
        from modeling.nnFormer.nnFormer_seg import nnFormer

        seg_net = nnFormer(
            input_channels=1,
            num_classes=args.num_classes,
            crop_size=args.rand_crop_size,         # 입력 해상도
            patch_size=[2, 4, 4],                  # token partition
            window_size=[8, 8, 6, 4],              # hierarchical window 크기
        )

    # -------------------------------------------------------
    # 6) TransBTS
    #    CNN + Transformer based 3D brain tumor segmentation model
    # -------------------------------------------------------
    elif args.method == "transbts":
        from modeling.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

        # TransBTS는 두 개의 네트워크를 반환하는 구조
        # 첫 번째는 encoder, 두 번째가 segmentation model → 두 번째만 사용
        _, seg_net = TransBTS(img_dim=args.rand_crop_size, num_classes=args.num_classes)

    # -------------------------------------------------------
    # 7) 정의되지 않은 모델 요청 시 에러 처리
    # -------------------------------------------------------
    else:
        raise NotImplementedError

    return seg_net
