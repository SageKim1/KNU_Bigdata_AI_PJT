import os
import cv2
import json
import yaml
import numpy as np
from glob import glob
from ultralytics import YOLO

# 설정 경로
TEST_IMG_DIR = r"D:\Project\PJT_07\test\images"  # test 이미지 폴더 경로
GT_LABEL_DIR = r"D:\Project\PJT_07\test\labels"  # GT 라벨 txt (YOLO 형식) 폴더
SAVE_ROOT = r"D:\Project\PJT_07\backup\test_result"  # 결과 저장 최상위 경로

MODEL_PATH = r"runs/detect/yolov11x_custom_003/weights/best.pt"
IOU_THRESHOLD = 0.5  # IoU 기준치
CONF_BANDS = [(0.9,1.0),(0.8,0.9),(0.7,0.8),(0.6,0.7),(0.0,0.6)]  # confidence 대역

# 클래스명 로드 (data.yaml 위치에 따라 경로 변경)
with open(r"D:\Project\PJT_07\code\data.yaml", 'r') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']

# 모델 로드
model = YOLO(MODEL_PATH)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def yolo_bbox_to_xyxy(bbox, img_w, img_h):
    x_c, y_c, w, h = bbox
    x1 = int((x_c - w/2)*img_w)
    y1 = int((y_c - h/2)*img_h)
    x2 = int((x_c + w/2)*img_w)
    y2 = int((y_c + h/2)*img_h)
    return [x1, y1, x2, y2]

def get_conf_band(conf):
    for low, high in CONF_BANDS:
        if low <= conf < high:
            return f"{low:.1f}-{high:.1f}"
    return "unknown"

def create_dirs(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

def draw_bbox(img, bbox, label, color, thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def load_gt_labels(gt_path, img_w, img_h):
    gt_boxes = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts=line.strip().split()
                cls = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                xyxy = yolo_bbox_to_xyxy(bbox, img_w, img_h)
                gt_boxes.append({'cls':cls, 'bbox':xyxy})
    return gt_boxes

def match_pred_gt(pred_box, pred_cls, gt_boxes, iou_thresh=IOU_THRESHOLD):
    best_iou = 0
    best_gt = None
    for gt in gt_boxes:
        iou = compute_iou(pred_box, gt['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
    if best_gt and (best_iou >= iou_thresh) and (pred_cls == best_gt['cls']):
        return 'match_class_iou_ok', best_iou, best_gt
    if best_gt and (pred_cls == best_gt['cls']) and (best_iou < iou_thresh):
        return 'match_class_iou_low', best_iou, best_gt
    return 'mismatch_class', best_iou, best_gt

def save_pred_json(save_path, pred_data):
    with open(save_path, 'w') as f:
        json.dump(pred_data, f, indent=2, ensure_ascii=False)

def main():
    img_paths = glob(os.path.join(TEST_IMG_DIR, '*.jpg'))
    total = len(img_paths)

    for idx, img_path in enumerate(img_paths, 1):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"🔄 [{idx}/{total}] 처리 중: {img_name} ...")

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # 추론
        results = model.predict(img, imgsz=640, conf=0.1, verbose=False)
        result = results[0]

        pred_boxes = []
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
            bbox = list(map(int, box))
            pred_boxes.append({'bbox': bbox, 'cls': int(cls), 'conf': float(conf)})

        gt_path = os.path.join(GT_LABEL_DIR, img_name+'.txt')
        gt_boxes = load_gt_labels(gt_path, img_w, img_h)

        # 예측 JSON 저장
        json_save_dir = os.path.join(SAVE_ROOT, 'json')
        create_dirs(json_save_dir)
        json_save_path = os.path.join(json_save_dir, f"{img_name}_pred.json")
        save_pred_json(json_save_path, pred_boxes)

        # 매칭 확인을 위한 GT 중복 방지 플래그 초기화
        gt_matched_flags = [False]*len(gt_boxes)

        # pred 별 매칭 및 이미지 저장
        for pred in pred_boxes:
            category = pred['cls']
            conf = pred['conf']
            bbox = pred['bbox']

            match_type, best_iou, best_gt = match_pred_gt(bbox, category, gt_boxes)
            conf_band = get_conf_band(conf)
            base_dir = os.path.join(SAVE_ROOT, match_type)

            pred_dir = os.path.join(base_dir, 'pred', f'class_{class_names[category]}', f'conf_{conf_band}')
            create_dirs(pred_dir)

            gt_pred_dir = os.path.join(base_dir, 'gt_pred_iou', f'class_{class_names[category]}', f'conf_{conf_band}')
            create_dirs(gt_pred_dir)

            img_pred = img.copy()
            img_gt_pred = img.copy()

            label = f"{class_names[category]} {conf:.2f}"
            draw_bbox(img_pred, bbox, label, (0,255,0))

            for i, gt in enumerate(gt_boxes):
                draw_bbox(img_gt_pred, gt['bbox'], class_names[gt['cls']], (255,0,0), thickness=2)
                # 매칭된 GT는 플래그 설정
                if best_gt is not None and gt == best_gt and best_iou >= IOU_THRESHOLD and category == gt['cls']:
                    gt_matched_flags[i] = True

            draw_bbox(img_gt_pred, bbox, label, (0,255,0), thickness=2)
            iou_text = f"IoU: {best_iou:.2f}"
            cv2.putText(img_gt_pred, iou_text, (bbox[0], bbox[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            save_name = f"{img_name}_P{conf:.2f}_I{best_iou:.2f}.jpg"
            cv2.imwrite(os.path.join(pred_dir, save_name), img_pred)
            cv2.imwrite(os.path.join(gt_pred_dir, save_name), img_gt_pred)

        # FN(매칭 안된 GT) 이미지 저장
        fn_base_dir = os.path.join(SAVE_ROOT, 'false_negative')
        for i, matched in enumerate(gt_matched_flags):
            if not matched:
                gt = gt_boxes[i]
                fn_class_dir = os.path.join(fn_base_dir, f'class_{class_names[gt["cls"]]}')
                create_dirs(fn_class_dir)
                img_fn = img.copy()
                label = f"FN: {class_names[gt['cls']]}"
                draw_bbox(img_fn, gt['bbox'], label, (0,0,255), thickness=3)
                save_name = f"{img_name}_FN_class{class_names[gt['cls']]}.jpg"
                cv2.imwrite(os.path.join(fn_class_dir, save_name), img_fn)

        print(f"✅ 완료: {img_name}\n")

if __name__ == "__main__":
    main()
