import yaml
import json
from pathlib import Path
from ultralytics import YOLO
import torch

from inference import get_max_angle_diff, get_mse, get_rmse

TRAIN_CONFIG = "configs/train.yaml"
INFERENCE_CONFIG = "configs/inference.yaml"

DATASET_CONFIGS_DIR = Path("dataset_configs")
DATASETS_DIR = Path("datasets")


def main():
    try:
        with open(TRAIN_CONFIG, "r") as cfg_file:
            train_cfg = yaml.safe_load(cfg_file)
        with open(INFERENCE_CONFIG, "r") as cfg_file:
            inference_cfg = yaml.safe_load(cfg_file)
    except yaml.YAMLError as e:
        print(e)

    # n, s, m, l, x
    # model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
    # model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # 3-fold cross validation
    project_base_dir = Path(train_cfg["project"])
    # project_base_dir.mkdir(parents=True, exist_ok=False)

    metrics = {
        "avg_mse": 0,
        "avg_rmse": 0,
        "fold01": {"mse": 0, "rmse": 0, "pred_label_pairs": []},
        "fold02": {"mse": 0, "rmse": 0, "pred_label_pairs": []},
        "fold03": {"mse": 0, "rmse": 0, "pred_label_pairs": []},
    }

    for id in range(3):
        fold_id = f"fold0{id + 1}"
        train_fold_dir = project_base_dir / fold_id
        train_cfg["data"] = DATASET_CONFIGS_DIR / f"{fold_id}.yaml"
        train_cfg["project"] = train_fold_dir
        model = YOLO(
            train_cfg["model"]
        )  # load a pretrained model (recommended for training)
        results = model.train(**train_cfg)
        del model
        torch.cuda.empty_cache()

        # assume we don't use duplicated names for training
        # inference_cfg["save"] = False
        inference_cfg["model"] = train_fold_dir / "train/weights/best.pt"
        inference_cfg["source"] = DATASETS_DIR / f"{fold_id}/val/images"
        inference_cfg["project"] = project_base_dir / fold_id
        model = YOLO(inference_cfg["model"])
        results = model.predict(**inference_cfg)
        del model
        torch.cuda.empty_cache()  # ok this shit still not work :)

        max_angle_diffs = []
        for result in results:
            coords = result.obb.xywhr.to("cpu").numpy().copy().tolist()

            # discard bounding boxes with small conf. value
            # coords = [y for x, y in zip(result.obb.conf, coords) if x >= 0.8]
            coords.sort(key=lambda x: x[1])  # sort by y coord
            angles = [x[4] for x in coords]
            max_angle_diffs.append(get_max_angle_diff(angles, coords)[0])

        # assume each image has a angle label file :p
        labels = []
        label_dir = DATASETS_DIR / f"{fold_id}/val/angles"
        for filename in sorted(Path(label_dir).iterdir()):
            with open(filename, "r") as file:
                labels.append(json.load(file)["diff"])

        print(f"----------------------Fold{id + 1}----------------------")
        for pred, label in zip(max_angle_diffs, labels):
            print("pred:", pred, "|", "label:", label)
            metrics[fold_id]["pred_label_pairs"].append([pred, label])

        metrics[fold_id]["mse"] = get_mse(max_angle_diffs, labels)
        metrics[fold_id]["rmse"] = get_rmse(max_angle_diffs, labels)
        print("mse:", metrics[fold_id]["mse"], "rmse:", metrics[fold_id]["rmse"])
        metrics["avg_mse"] += metrics[fold_id]["mse"]
        metrics["avg_rmse"] += metrics[fold_id]["rmse"]

    metrics["avg_mse"] /= 3
    metrics["avg_rmse"] /= 3

    with open(project_base_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


if __name__ == "__main__":
    main()
