import os
from glob import glob
import yaml
import csv
from collections import Counter

# 설정 경로
DATA_YAML_PATH = r"D:\Project\PJT_07\code\data.yaml"
DATA_SPLITS = {
    'train': r"D:\Project\PJT_07\train",
    'valid': r"D:\Project\PJT_07\valid",
    'test': r"D:\Project\PJT_07\test",
}
SAVE_DIR = r"D:\Project\PJT_07\code\class_counts"  # CSV 저장 경로

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def save_csv(split_name, counter, class_names):
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"class_count_{split_name}.csv")
    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class ID", "Class Name", "Count"])
        for cls_id in range(len(class_names)):
            writer.writerow([cls_id, class_names[cls_id], counter.get(cls_id, 0)])
    print(f"💾 CSV 저장 완료: {save_path}")

def count_classes(data_splits, class_names):
    for split_name, split_dir in data_splits.items():
        label_dir = os.path.join(split_dir, 'labels')
        label_files = glob(os.path.join(label_dir, '*.txt'))

        counter = Counter()

        for label_path in label_files:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(class_names):
                            counter[cls_id] += 1
                        else:
                            print(f"⚠️ {label_path}: Invalid class ID {cls_id}")
                    except ValueError:
                        print(f"⚠️ {label_path}: Malformed line -> {line.strip()}")

        print(f"\n📊 클래스별 객체 수 ({split_name.upper()}):")
        for cls_id, count in sorted(counter.items()):
            print(f"  {cls_id}: {class_names[cls_id]} -> {count} 개")
        if not counter:
            print("  (라벨 데이터 없음)")

        # CSV 저장
        save_csv(split_name, counter, class_names)

if __name__ == "__main__":
    class_names = load_class_names(DATA_YAML_PATH)
    count_classes(DATA_SPLITS, class_names)
