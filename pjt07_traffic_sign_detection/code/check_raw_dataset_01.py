import os
from glob import glob
import yaml

# 설정 경로 (필요에 따라 수정)
DATA_YAML_PATH = r"D:\Project\PJT_07\code\data.yaml"

DATA_SPLITS = {
    'train': r"D:\Project\PJT_07\train",
    'valid': r"D:\Project\PJT_07\valid",
    'test': r"D:\Project\PJT_07\test",
}

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def check_files_and_classes(data_splits, class_names):
    max_class_id = len(class_names) - 1
    errors_found = False

    for split_name, split_dir in data_splits.items():
        print(f"\n--- Checking split: {split_name} ---")

        img_dir = os.path.join(split_dir, 'images')
        label_dir = os.path.join(split_dir, 'labels')

        img_files = set(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(img_dir, '*.*')))
        label_files = set(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(label_dir, '*.txt')))

        # 1. 이미지와 라벨 쌍 존재 여부 체크
        imgs_without_label = img_files - label_files
        labels_without_img = label_files - img_files

        if imgs_without_label:
            errors_found = True
            print(f"  [!] Images without GT labels ({len(imgs_without_label)}): {sorted(imgs_without_label)}")
        else:
            print("  All images have corresponding GT labels.")

        if labels_without_img:
            errors_found = True
            print(f"  [!] GT labels without images ({len(labels_without_img)}): {sorted(labels_without_img)}")
        else:
            print("  All GT labels have corresponding images.")

        # 2. 라벨 내 클래스 ID 체크
        invalid_class_ids = set()
        for label_name in label_files:
            label_path = os.path.join(label_dir, label_name + '.txt')
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                    except:
                        print(f"  [!] Invalid format in {label_path} line {line_num}")
                        errors_found = True
                        continue
                    if cls_id < 0 or cls_id > max_class_id:
                        invalid_class_ids.add((label_name, cls_id, line_num))

        if invalid_class_ids:
            errors_found = True
            print(f"  [!] Found class IDs outside defined range 0~{max_class_id}:")
            for label_name, cls_id, line_num in sorted(invalid_class_ids):
                print(f"     - File '{label_name}.txt', line {line_num}: class_id={cls_id}")
        else:
            print("  All GT labels have valid class IDs.")

    if not errors_found:
        print("\n✅ 모든 데이터셋 점검 완료: 이상 없음.")

if __name__ == "__main__":
    class_names = load_class_names(DATA_YAML_PATH)
    check_files_and_classes(DATA_SPLITS, class_names)
