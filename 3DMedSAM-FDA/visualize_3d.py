# pip install pyvista nibabel numpy
"""
3D 의료 영상 및 예측 마스크 시각화
실행하면 검은색 창이 뜨며 3D 뷰어가 실행됩니다.
마우스 왼쪽 드래그: 회전
마우스 휠: 확대/축소
Shift + 드래그: 이동
"""
# img_file = "/workspace/dataset/kist_update/data/case_00000/imaging.nii.gz"
# mask_file = "/workspace/dataset/kist_update/data/case_00000/segmentations/tumor_instance-1_annotation-1.nii.gz"
"""
libGL.so 관련 에러가 뜬다면?
apt-get update
apt-get install -y libgl1-mesa-glx xvfb
"""
"""
Usage example:
python visualize_3d.py
"""

import pyvista as pv
import nibabel as nib
import numpy as np
import os

# [핵심 1] 가상 디스플레이 시작 (Segmentation Fault 방지)
# 이 코드가 없으면 서버에서 100% 터집니다.
pv.start_xvfb()

# 테마 설정
pv.set_plot_theme("dark")

def save_3d_visualization(image_path, mask_path, output_file="result_3d.png"):
    print(f"Loading data...\n Image: {image_path}\n Mask:  {mask_path}")
    
    try:
        img_obj = nib.load(image_path)
        mask_obj = nib.load(mask_path)
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    img_data = img_obj.get_fdata()
    mask_data = mask_obj.get_fdata()
    spacing = img_obj.header.get_zooms()

    # [핵심 2] UniformGrid -> ImageData 로 변경 (버전 호환성 해결)
    grid = pv.ImageData()
    grid.dimensions = img_data.shape
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.point_data["Intensity"] = img_data.flatten(order="F")
    grid.point_data["Mask"] = mask_data.flatten(order="F")

    # [핵심 3] off_screen=True 필수
    p = pv.Plotter(off_screen=True, window_size=[1000, 1000])
    
    # 1. CT 영상 단면 추가
    slices = grid.slice_orthogonal()
    p.add_mesh(slices, scalars="Intensity", cmap="gray", opacity=1.0, show_scalar_bar=False)

    # 2. 종양 3D 입체 추가
    if mask_data.max() > 0:
        # contour 생성 시 에러가 난다면 method='marching_cubes' 등 옵션 조정 가능
        mask_mesh = grid.contour(isosurfaces=[0.5], scalars="Mask")
        p.add_mesh(mask_mesh, color="red", opacity=0.7, show_edges=False)
        print("✅ Tumor mask found and added.")
    else:
        print("⚠️ Warning: Mask is empty.")

    p.camera_position = 'iso'
    p.add_text("3D Visualization", position='upper_left', font_size=10, color='white')

    print(f"Saving visualization to {output_file}...")
    p.screenshot(output_file)
    print("Done! 🎉")

if __name__ == "__main__":
    img_file = "/workspace/dataset/kist_update/data/case_00000/imaging.nii.gz"
    mask_file = "/workspace/dataset/kist_update/data/case_00000/segmentations/tumor_instance-1_annotation-1.nii.gz"
    
    save_3d_visualization(img_file, mask_file)
