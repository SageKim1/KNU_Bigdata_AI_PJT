import os
import json
import csv
from ultralytics import YOLO

def run_evaluation():
    model_paths = [
        r"D:\Project\PJT_07\code\runs\detect\yolov11n_custom_001\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11n_custom_002\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11n_custom_003\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11n_custom_004\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11n_custom_005\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11n_custom_006\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11x_custom_001\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11x_custom_002\weights\best.pt",
        r"D:\Project\PJT_07\code\runs\detect\yolov11x_custom_003\weights\best.pt"
    ]

    save_dir = r"D:\Project\PJT_07\backup\test_result_map_summary"
    os.makedirs(save_dir, exist_ok=True)

    results_summary = {}

    for model_path in model_paths:
        print(f"🚀 검증 시작: {model_path}")
        model = YOLO(model_path)
        results = model.val(data='data.yaml', split='test')

        map50 = results.box.map50.item()   # mAP@0.5
        map5095 = results.box.map.item()   # mAP@0.5:0.95
        print(f"mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}")

        model_name = model_path.split('\\')[-3]
        results_summary[model_name] = {
            "model_path": model_path,
            "mAP_0.5": map50,
            "mAP_0.5:0.95": map5095
        }

    # JSON 저장
    json_path = os.path.join(save_dir, "test_map_results.json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(results_summary, jf, indent=2, ensure_ascii=False)
    print(f"📄 JSON 저장 완료: {json_path}")

    # CSV 저장
    csv_path = os.path.join(save_dir, "test_map_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["model_name", "model_path", "mAP@0.5", "mAP@0.5:0.95"])
        for model_name, metrics in results_summary.items():
            writer.writerow([
                model_name,
                metrics["model_path"],
                f"{metrics['mAP_0.5']:.4f}",
                f"{metrics['mAP_0.5:0.95']:.4f}"
            ])
    print(f"📄 CSV 저장 완료: {csv_path}")

if __name__ == '__main__':
    run_evaluation()
