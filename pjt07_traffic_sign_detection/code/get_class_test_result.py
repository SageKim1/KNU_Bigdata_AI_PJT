import os
import csv
from ultralytics import YOLO

def evaluate_and_save():
    model_path = r"D:\Project\PJT_07\code\runs\detect\yolov11x_custom_003\weights\best.pt"
    data_yaml = r"data.yaml"  # data.yaml 경로 (test셋 split 포함)

    save_dir = r"D:\Project\PJT_07\backup\test_result_map_summary"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "yolov11x_custom_003_test_class_metrics.csv")

    model = YOLO(model_path)

    # test셋 기준 평가 수행 (split='test')
    results = model.val(data=data_yaml, split='test', batch=4, device='0')  # 배치 크기를 4로 제한

    # 클래스별 metrics 딕셔너리
    cls_metrics = results.box.cls  # dict-like
    class_names = model.names

    # 클래스별 정보들: Images, Instances, Box Precision(P), Recall(R), mAP50, mAP50-95
    # results.box.cls 내부 키: 'images', 'instances', 'P', 'R', 'map50', 'map'
    images_per_class = cls_metrics.images.cpu().numpy() if hasattr(cls_metrics.images, 'cpu') else cls_metrics.images
    instances_per_class = cls_metrics.instances.cpu().numpy() if hasattr(cls_metrics.instances, 'cpu') else cls_metrics.instances
    precision_per_class = cls_metrics.P.cpu().numpy() if hasattr(cls_metrics.P, 'cpu') else cls_metrics.P
    recall_per_class = cls_metrics.R.cpu().numpy() if hasattr(cls_metrics.R, 'cpu') else cls_metrics.R
    map50_per_class = cls_metrics.map50.cpu().numpy() if hasattr(cls_metrics.map50, 'cpu') else cls_metrics.map50
    map5095_per_class = cls_metrics.map.cpu().numpy() if hasattr(cls_metrics.map, 'cpu') else cls_metrics.map

    # 출력
    print(f"모델: {model_path}")
    print(f"\n{'Class':15s} {'Images':7s} {'Instances':9s} {'Box(P)':9s} {'Recall':8s} {'mAP50':8s} {'mAP50-95':9s}")
    print("-" * 70)
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name:15s} {int(images_per_class[i]):7d} {int(instances_per_class[i]):9d} "
              f"{precision_per_class[i]:9.3f} {recall_per_class[i]:8.3f} "
              f"{map50_per_class[i]:8.3f} {map5095_per_class[i]:9.3f}")

    # CSV 저장
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Images', 'Instances', 'Box(P)', 'Recall', 'mAP50', 'mAP50-95'])
        for i, cls_name in enumerate(class_names):
            writer.writerow([
                cls_name,
                int(images_per_class[i]),
                int(instances_per_class[i]),
                f"{precision_per_class[i]:.3f}",
                f"{recall_per_class[i]:.3f}",
                f"{map50_per_class[i]:.3f}",
                f"{map5095_per_class[i]:.3f}"
            ])

    print(f"\n✅ CSV 저장 완료: {csv_path}")

if __name__ == "__main__":
    evaluate_and_save()
