import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFFTPath3D(nn.Module):
    """
    Local Path with 3D FFT-based High-Frequency Enhancement
    (Includes Flat-Region Skipping to prevent NaN)
    """
    def __init__(
        self,
        channels: int,
        high_freq_boost: float = 0.5,
        min_radius_ratio: float = 0.25,    
    ):
        super().__init__()
        self.high_freq_boost = high_freq_boost
        self.min_radius_ratio = min_radius_ratio

        self.post_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(channels, affine=True, eps=1e-5) # eps 확인
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [디버깅] 입력부터 이미 NaN인지 확인
        if torch.isnan(x).any() or torch.isinf(x).any():
            # print("Warning: Input to LocalFFTPath3D contains NaN/Inf. Replacing with 0.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # [핵심] 평탄한 영역(모든 픽셀값이 비슷함) 체크
        # 표준편차가 0에 가까우면 정규화나 FFT 시 수치 불안정 발생 -> Skip
        if x.std() < 1e-4:
            return torch.zeros_like(x)

        B, C, D, H, W = x.shape
        original_dtype = x.dtype
        x_fp32 = x.float()

        # 1) FFT (norm='ortho')
        x_fft = torch.fft.fftn(x_fp32, dim=(-3, -2, -1), norm='ortho')

        # 2) 주파수 그리드 생성
        device = x.device
        freq_d = torch.fft.fftfreq(D, device=device).view(1, 1, D, 1, 1)
        freq_h = torch.fft.fftfreq(H, device=device).view(1, 1, 1, H, 1)
        freq_w = torch.fft.fftfreq(W, device=device).view(1, 1, 1, 1, W)

        radius = torch.sqrt(freq_d ** 2 + freq_h ** 2 + freq_w ** 2)
        radius_max = radius.max().clamp(min=1e-6)
        radius_norm = radius / radius_max

        # 3) 마스크 생성 (Clamp 적용)
        r0 = self.min_radius_ratio
        high_region = torch.clamp(radius_norm - r0, min=0.0) / (1.0 - r0 + 1e-6)
        mask_val = 1.0 + self.high_freq_boost * high_region
        mask = torch.clamp(mask_val, max=2.0) 

        # 4) 주파수 강조
        x_fft_hf = x_fft * mask

        # 5) IFFT
        x_hf = torch.fft.ifftn(x_fft_hf, dim=(-3, -2, -1), norm='ortho').real

        # [안전장치] 결과값 폭발 방지
        if torch.isnan(x_hf).any() or torch.isinf(x_hf).any():
            # print("Warning: NaN detected after IFFT. Zeroing out.")
            x_hf = torch.zeros_like(x_hf)
        
        x_hf = torch.clamp(x_hf, min=-5.0, max=5.0)

        # 6) 복구 및 후처리
        x_hf = x_hf.to(dtype=original_dtype)
        
        # Conv -> Norm -> Act 과정에서도 NaN 발생 가능하므로 체크
        try:
            x_hf = self.post_conv(x_hf)
            x_hf = self.norm(x_hf)
            x_hf = self.act(x_hf)
        except RuntimeError:
            return torch.zeros_like(x)

        return x_hf

class Adapter(nn.Module):
    """
    Dual-Path 3D Adapter with FFT-based Local Path

    - Input / Output: features (B, D, H, W, C)
      C = input_dim

    Paths:
      (1) Global 3D Context Path: 3x3x3 depthwise conv (저주파 / 형태 정보)
      (2) Local FFT Path: 3D FFT로 고주파 강조 (텍스처 / 경계 정보)
      (3) Gated Fusion: global → gate → local_fft 조절
    """

    def __init__(
        self,
        input_dim: int,
        mid_dim: int
    ):
        super().__init__()

        # 공통: token 차원 -> mid_dim
        self.linear1 = nn.Linear(input_dim, mid_dim)

        # (1) Global 3D Context Path
        self.global_conv = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=3,
            padding=1,
            groups=mid_dim,   # depthwise 3D conv
        )

        # (2) Local FFT Path
        self.local_fft = LocalFFTPath3D(
            channels=mid_dim,
            # high_freq_boost=2.0,
            high_freq_boost=0.5,
            min_radius_ratio=0.25,
        )

        # (3) Fusion: Global → Gate
        self.fusion_gate = nn.Conv3d(
            in_channels=mid_dim,
            out_channels=mid_dim,
            kernel_size=1,
        )

        # 최종 proj: mid_dim -> input_dim
        self.linear2 = nn.Linear(mid_dim, input_dim)
        # 학습 초기에 어댑터가 노이즈를 생성하지 않도록 가중치를 0으로 설정 # loss nan 발산 방지 
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D, H, W, C)
        """
        # 0) token dim 축소
        x = self.linear1(features)          # (B, D, H, W, mid_dim)
        x = F.relu(x)

        # Conv3d형태로: (B, C, D, H, W)
        x_3d = x.permute(0, 4, 1, 2, 3)

        # ============================
        # (1) Global 3D Context Path
        # ============================
        global_feat = self.global_conv(x_3d)   # (B, mid_dim, D, H, W)
        global_feat = F.relu(global_feat)

        # ============================
        # (2) Local FFT Path
        # ============================
        local_feat = self.local_fft(x_3d)      # (B, mid_dim, D, H, W)

        # ============================
        # (3) Gated Fusion
        # ============================
        gate = torch.sigmoid(self.fusion_gate(global_feat))  # (B, mid_dim, D, H, W)
        fused_3d = global_feat + gate * local_feat           # (B, mid_dim, D, H, W)

        # 다시 (B, D, H, W, mid_dim)로
        fused = fused_3d.permute(0, 2, 3, 4, 1)

        # 최종 proj + residual
        out = self.linear2(fused)
        out = F.relu(out)

        out = features + out
        return out
