# Oriented-Bounding-Box Cobb Angle Measurement

## About The Project
Fine-tuning yolo11*-obb models for Cobb angle measurement.

## Built With
- YOLOv11

## Getting Started
### Prerequisites
- Python 3.12.7+

### Installation
1. Clone the repo
```sh
git clone https://github.com/dereklee0310/OCAM
   ```
2. Install dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Training
1. Open `configs/train.yaml`
2. Edit hyperparameters and project name (save dir)
```sh
python tools/train.py
```

### Inference
1. Open `configs/inference.yaml`
2. Edit source (input image path or directory)
```sh
python tools/inference.py
```

### Demo
```sh
python tools/demo.py
```

## Contributing
PR or bug report are welcome.

## License
TBD

## Contact
dereklee0310@gmail.com

## Acknowledgments
- [@zhejia14](https://github.com/zhejia14)
- NCKU SIVS Lab
- [ultralytics](https://github.com/ultralytics/ultralytics)

## MISC
Weights are too large so I didn't upload them :p