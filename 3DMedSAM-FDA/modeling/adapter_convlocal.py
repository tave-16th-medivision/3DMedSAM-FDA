import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """
    Dual-Path 3D Adapter
    - Input : features (B, D, H, W, C)
    - Output: features (B, D, H, W, C) + residual
    """

    def __init__(
        self,
        input_dim: int,
        mid_dim: int
    ):
        super().__init__()

        # 공통: token 차원 -> mid_dim (경로 1, 2 공용 인코딩)
        self.linear1 = nn.Linear(input_dim, mid_dim)

        # (1) Global 3D Context Path
        #  - 기존 adapter와 유사한 3D depthwise conv
        self.global_conv = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=3,
            padding=1,
            groups=mid_dim,    # depthwise
        )

        # (2) Local 2.5D Texture Path
        #  - 먼저 슬라이스별 2D 텍스처 강화: (1, 3, 3) 커널 (in-plane)
        self.local_conv_2d = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=mid_dim,    # 각 채널 독립 2D conv
        )
        #  - 그 다음 제한적 3D 맥락: (3, 1, 1) 커널 (depth 방향)
        self.local_conv_3d = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=mid_dim,    # 각 채널 독립 z-context
        )

        # (3) Intelligent Fusion (Gated Fusion)
        #  - Global feature로부터 gate 생성 (spatial + channel-wise)
        self.fusion_gate = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=1,
        )

        # 출력: mid_dim -> input_dim, residual 연결
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D, H, W, C)
        """

        # 공통 인코딩: token dim -> mid_dim
        # shape: (B, D, H, W, mid_dim)
        x = self.linear1(features)
        x = F.relu(x)

        # Conv3d 입력 형식으로 변환: (B, C, D, H, W)
        x_3d = x.permute(0, 4, 1, 2, 3)

        # ============================
        # (1) Global 3D Context Path
        # ============================
        # 3x3x3 depthwise conv로 global shape / low-freq 잡기
        global_feat = self.global_conv(x_3d)
        global_feat = F.relu(global_feat)

        # ============================
        # (2) Local 2.5D Texture Path
        # ============================
        # 2D(in-plane) conv: (1,3,3) -> 슬라이스별 텍스처 강화
        local_feat = self.local_conv_2d(x_3d)
        local_feat = F.relu(local_feat)

        # 제한적 3D conv: (3,1,1) -> z-context만 살짝 추가
        local_feat = self.local_conv_3d(local_feat)
        local_feat = F.relu(local_feat)

        # ============================
        # (3) Intelligent Gated Fusion
        # ============================
        # Global feature로부터 gate 생성 (0~1)
        gate = torch.sigmoid(self.fusion_gate(global_feat))
        # gate * local: "여기가 중요하니 local texture를 더 통과시켜라"
        fused_3d = global_feat + gate * local_feat  # (B, mid_dim, D, H, W)

        # 다시 (B, D, H, W, mid_dim)로
        fused = fused_3d.permute(0, 2, 3, 4, 1)

        # 최종 proj + residual
        out = self.linear2(fused)
        out = F.relu(out)

        # residual 연결 (원래 token + adapter output)
        out = features + out
        return out
