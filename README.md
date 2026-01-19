# 3DMedSAM-FDA
: A Frequency-based Dual-Path Adapter for 3D Medical Image Segmentation

**3DMedSAM-FDA**는 기존 [3D SAM Adapter](https://github.com/med-air/3DSAM-adapter)가 전역적인 의미 정보에 비해 국소 구조와 경계 정보를 충분히 반영하지 못하는 한계를 극복하기 위해 제안되었습니다. 우리는 전역 문맥 정보와 주파수 기반의 국소 정보를 병렬적으로 처리하는 새로운 이중 경로 어댑터(Dual-Path Adapter)를 소개합니다.

## Abstract
프롬프트 기반 분할 모델([Segment Anything Model, SAM](https://github.com/facebookresearch/segment-anything))을 3차원 의료 영상에 적용하려는 시도가 늘어나고 있습니다. 하지만 기존 3D 어댑터 구조는 종양과 같이 크기가 작고 경계가 불명확한 병변을 정밀하게 분할하는 데 한계가 있습니다. 

3DMedSAM-FDA는 다음과 같은 특징을 가집니다:
- 이중 경로 구조 (Dual-Path Architecture): 전역 문맥 경로(Global Context Path)와 국소 경로(Local Path)를 병렬로 결합했습니다.
- 주파수 도메인 활용 (Frequency Domain Analysis): 국소 경로에서 3D FFT를 통해 고주파 성분을 선택적으로 강조하여 경계 및 미세 구조 정보를 보강합니다.
- 게이트 융합 (Gated Fusion): 전역 특징으로부터 생성된 게이트를 통해 국소 특징의 기여도를 조절하여 두 정보를 효과적으로 통합합니다.

## Architecture
전체 분할 모델의 프레임워크는 이미지 인코더, 프롬프트 인코더, 마스크 디코더로 구성되며, 사전학습된 SAM(Segment Anything Model)을 고정(Frozen)한 상태에서 Image Encoder에 간단한 Adapter를 삽입하여 미세조정하는 [3DSAM-Adapter](https://github.com/med-air/3DSAM-adapter)의 방식을 따릅니다. 우리 프레임워크는 기존의 단순한 단일 경로 어댑터가 아닌, Global Path와 Local Path로 구성된 Dual-path Adapter 방식을 사용합니다. 자세한 구조는 아래와 같습니다: 
<p align="center">
<img width="700" alt="framework" src="https://github.com/user-attachments/assets/c6aa6836-5164-4cdb-bf3d-193921694538" />
</p> 

우리가 제안하는 어댑터(Locality-enhanced, Frequency-based Dual-path Adapter)는 다음과 같이 동작합니다:
1. Global Context Path (Spatial Domain): $3 \times 3 \times 3$ Depth-wise Conv를 사용하여 3차원 형태와 장기 문맥 정보를 포착합니다
3. Local Textural Path (Frequency Domain):
   - 입력 특징을 주파수 도메인으로 변환 (FFT).
   - 반경 기반 마스크 $M(r)$를 적용하여 고주파 성분(경계 정보) 증폭.
   - 공간 도메인으로 복원 (IFFT) 후 $1 \times 1 \times 1$ Conv 적용.
3. Adaptive Gated Fusion: 전역 경로의 정보를 바탕으로 국소 경로 정보의 반영 비율을 조정하여 결합합니다.  

## Results
실험은 3차원 CT 영상에 대한 4가지 복부 종양(Kidney/Pancreas/Liver/Colon Tumor) 분할 작업을 수행합니다. 데이터셋 및 데이터 전처리, 기본 학습 하이퍼파라미터 설정 등의 자세한 실험 세팅은 [3DSAM-Adapter](https://github.com/med-air/3DSAM-adapter)의 방식을 그대로 따릅니다. 

<p align="center">
<img width="700" alt="실험결과정량" src="https://github.com/user-attachments/assets/2f187f74-0545-4514-83fc-252112b7a43d" />
</p>

실험 결과, 제안한 3DMedSAM-FDA는 장기별 특성에 따라 기존 방법 대비 일관되고 유의미한 성능 향상을 보였습니다.
신장암(Kidney Tumor) 및 췌장암(Pancreas Tumor) 분할에서는 기존 3D SAM Adapter 대비 평균적으로 각각 +5.1%p, +0.1%p의 Dice 성능 향상을 달성하면서도, 전역 의미 정보를 유지하여 안정적인 분할 결과를 확인하였습니다.

특히 간암(Liver Tumor) 분할에서는 포인트 수가 증가할수록 성능 향상이 두드러지게 나타났으며, 10-point 설정 기준으로 기존 3D SAM Adapter뿐만 아니라 nnU-Net 대비서도 1.87%p 높은 Dice 점수를 기록하였습니다. 이는 전역 문맥 정보와 국소 구조 정보를 효과적으로 결합한 이중 경로 구조의 강점을 보여줍니다. 

결장암 분할(Colon Tumor)에서는 모든 포인트 설정에서 기존 방법 대비 큰 폭의 성능 개선을 달성하였습니다. 기존 모델들이 충분한 성능을 보이지 못한 복잡하고 작은 병변 환경에서도, 제안한 방법은 주파수 기반 국소 경로를 통한 경계 및 미세 구조 보강을 통해 분할 성능을 안정적으로 향상시켰습니다.

뿐만 아니라, 제안한 방법은 기존 3D SAM Adapter(25.46M)와 유사한 수준의 파라미터 수(29.02M)를 유지하면서도 이러한 성능 향상을 달성하여, 경량화된 구조 하에서도 표현력을 효과적으로 확장할 수 있음을 입증하였습니다.

<p align="center">
<img width="600" alt="실험결과정성" src="https://github.com/user-attachments/assets/954407b0-b61e-478e-a5fa-ed916d98c153" />
</p>

위는 정성적 실험 결과로, 특히 Liver Tumor, Colon Tumor와 같은 미세한 질감 및 경계가 중요한 모달리티에서 기존 3D SAM Adapter 대비 우수한 성능을 보이는 것을 시각화합니다. 

---

## Others

### Model Checkpoints
학습한 모델 가중치 파일 및 로그는 github release를 통해 확인할 수 있습니다.

### Installation
본 프로젝트는 Python 3.8+, PyTorch, CUDA 환경을 기반으로 합니다. 
실험은 모두 NVIDIA A40 GPU 1x 환경에서 진행되었습니다. (주의: FFT based Dual-path Adapter는 A100에서 학습이 불안정할 수 있습니다.)
GPU 환경은 [Vast.ai](https://vast.ai/)라는 GPU 서버 인스턴스 대여 플랫폼을 활용하였습니다. (`vastai-guide.md`의 사용법 상세 내용 참고.) 
아래 명령어를 통해 필요한 환경을 구성하고 의존성을 설치할 수 있습니다. 

Conda Environment (Recommended)
```
# 1. Create and activate conda environment
conda create -y -n med_sam python=3.9.16 
conda activate med_sam

# 2. Install PyTorch & Torchvision (CUDA 11.3)
# 환경에 맞게 설치 필요 
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 

# 3. Install Segment Anything & Surface Distance
pip install git+https://github.com/facebookresearch/segment-anything.git 
pip install git+https://github.com/deepmind/surface-distance.git 

# 4. Install remaining dependencies
pip install -r requirements.txt 
```

### Data Preparation
실험에 사용된 데이터셋(KiTS 2021, MSD-Pancreas, LiTS 2017, MSD-Colon)의 전처리 및 디렉토리 구조는 [3DSAM-Adapter](https://github.com/med-air/3DSAM-adapter)의 가이드를 따릅니다. 

### Usage
#### 1. Train
모델 학습을 위해 `3DMedSAM-FDA/train.py`를 실행합니다.
```
python train.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/" --max_epoch 200 
```

#### 2. Inference (Test) 
학습된 모델을 평가하기 위해 `3DMedSAM-FDA/test.py`를 실행합니다.
```
python test.py --data kits --snapshot_path "path/to/snapshot/" --data_prefix "path/to/data folder/"  --num_prompts 1 
```

#### 3. Detailed Experiments 
이외의 여러 세부적인 실험(모델 파라미터 분석, 2D/3D 시각화, 단일 케이스 추론 등)은 `notebooks/experiment.ipynb`에 사용 방법이 정리되어 있습니다. 

### Project Structure
```
3DMedSAM-FDA/
├── modeling/
│   ├── image_encoder.py       # Simply Modified Image Encoder with Adapter import 
│   ├── mask_decoder.py
│   ├── prompt_encoder.py
│   ├── adapter.py             # Original Adapter
│   ├── adapter_convlocal.py   # Spatial(Conv) Path Implementation
│   ├── adapter_fftlocal.py    # Frequency(FFT) Path Implementation (Proposed)
│   └── ... 
├── dataset/                   # Dataloader
├── utils/
├── train.py                   # Training Script
├── test.py                    # Inference Script
└── ... 
```

### Acknowledgements
이 프로젝트는 [3DSAM-Adapter](https://github.com/med-air/3DSAM-adapter)와 [Segment Anything](https://github.com/facebookresearch/segment-anything)을 기반으로 구축되었습니다. 
훌륭한 연구와 코드를 공개해 주신 저자분들께 감사드립니다.
