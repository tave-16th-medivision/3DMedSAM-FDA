# 3DMedSAM-FDA
A Frequency-based Dual-Path Adapter for 3D Medical Image Segmentation
3DMedSAM-FDA는 기존 3D SAM Adapter가 전역적인 의미 정보에 비해 국소 구조와 경계 정보를 충분히 반영하지 못하는 한계를 극복하기 위해 제안되었습니다. 우리는 전역 문맥 정보와 주파수 기반의 국소 정보를 병렬적으로 처리하는 새로운 이중 경로 어댑터(Dual-Path Adapter)를 소개합니다.

## Abstract
프롬프트 기반 분할 모델(Segment Anything Model, SAM)을 3차원 의료 영상에 적용하려는 시도가 늘어나고 있습니다. 하지만 기존 3D 어댑터 구조는 종양과 같이 크기가 작고 경계가 불명확한 병변을 정밀하게 분할하는 데 한계가 있습니다.

3DMedSAM-FDA는 다음과 같은 특징을 가집니다:
- 이중 경로 구조 (Dual-Path Architecture): 전역 문맥 경로(Global Context Path)와 국소 경로(Local Path)를 병렬로 결합했습니다.
- 주파수 도메인 활용 (Frequency Domain Analysis): 국소 경로에서 3D FFT를 통해 고주파 성분을 선택적으로 강조하여 경계 및 미세 구조 정보를 보강합니다.
- 게이트 융합 (Gated Fusion): 전역 특징으로부터 생성된 게이트를 통해 국소 특징의 기여도를 조절하여 두 정보를 효과적으로 통합합니다.
