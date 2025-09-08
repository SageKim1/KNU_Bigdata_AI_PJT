import os
import cv2
import yaml
from glob import glob

# 설정 경로
TEST_IMG_DIR = r"D:\Project\PJT_07\test\images"
GT_LABEL_DIR = r"D:\Project\PJT_07\test\labels"
SAVE_ROOT = r"D:\Project\PJT_07\backup\test_result\ground_truth"

# 클래스명 로드
with open(r"D:\Project\PJT_07\code\data.yaml", 'r') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def yolo_bbox_to_xyxy(bbox, img_w, img_h):
    x_c, y_c, w, h = bbox
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return [x1, y1, x2, y2]

def draw_bbox(img, bbox, label, color=(255, 0, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def load_gt_labels(gt_path, img_w, img_h):
    gt_boxes = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                xyxy = yolo_bbox_to_xyxy(bbox, img_w, img_h)
                gt_boxes.append({'cls': cls, 'bbox': xyxy})
    return gt_boxes

def main():
    img_paths = glob(os.path.join(TEST_IMG_DIR, '*.jpg'))
    total = len(img_paths)

    for idx, img_path in enumerate(img_paths, 1):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"🔄 [{idx}/{total}] 처리 중: {img_name} ...")

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        gt_path = os.path.join(GT_LABEL_DIR, img_name + '.txt')
        gt_boxes = load_gt_labels(gt_path, img_w, img_h)

        class_done = set()

        for gt in gt_boxes:
            cls_name = class_names[gt['cls']]
            if cls_name not in class_done:
                save_dir = os.path.join(SAVE_ROOT, cls_name)
                create_dir(save_dir)
                class_done.add(cls_name)

            # 원본 복사 후 클래스별 그리기
            img_copy = img.copy()
            # 이 클래스 박스만 그리기
            for box in [b for b in gt_boxes if class_names[b['cls']] == cls_name]:
                draw_bbox(img_copy, box['bbox'], class_names[box['cls']], color=(255, 0, 0), thickness=2)

            save_path = os.path.join(SAVE_ROOT, cls_name, f"{img_name}.jpg")
            cv2.imwrite(save_path, img_copy)
            print(f"✅ GT 저장 완료: {save_path}")

if __name__ == "__main__":
    main()
