from ultralytics import YOLO
from config.config import get_cfg_defaults

TRAINING_CONFIG = "configs/fold1.yaml"
PROJECT_DIR = "runs/obb/m_mosaic0_batch8_epoch50__"
PRETRAINED_WEIGHTS = "pretrained_weights/yolo11m-obb.pt"


def setup_cfg(args):
    """Merge args into cfg node."""
    cfg = get_cfg_defaults()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    return cfg.clone()

def main():
    # n, s, m, l, x
    # model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
    # model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO(PRETRAINED_WEIGHTS)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=TRAINING_CONFIG,
        project=PROJECT_DIR,
        mosaic=0,
        batch=8,
        epochs=4,
        imgsz=1200,
    )

if __name__ == "__main__":
    main()
