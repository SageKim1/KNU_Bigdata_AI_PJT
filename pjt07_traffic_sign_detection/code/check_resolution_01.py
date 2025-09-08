import os
import cv2
from glob import glob

# 데이터셋 이미지 폴더 경로
DATASET_ROOT = r"D:\Project\PJT_07"  # 필요 시 경로 수정
SPLITS = ['train', 'valid', 'test']

def get_image_paths(split_root):
    img_dir = os.path.join(split_root, 'images')
    exts = ('*.jpg', '*.jpeg', '*.png')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob(os.path.join(img_dir, ext)))
    return img_paths

def check_image_sizes(target_w=416, target_h=416):
    all_ok = True

    for split in SPLITS:
        split_root = os.path.join(DATASET_ROOT, split)
        img_paths = get_image_paths(split_root)
        print(f"🔍 {split}: 이미지 {len(img_paths)}개 확인 중...")

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 이미지 로드 실패: {img_path}")
                all_ok = False
                continue
            h, w = img.shape[:2]
            if w != target_w or h != target_h:
                print(f"❌ 해상도 불일치: {img_path} (width: {w}, height: {h})")
                all_ok = False

    if all_ok:
        print("\n✅ 모든 이미지가 416x416 입니다.")
    else:
        print("\n⚠️ 해상도가 다른 이미지가 있습니다.")

if __name__ == "__main__":
    check_image_sizes()
