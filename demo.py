import gradio as gr
import torch
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage.io import imread
import os

sys.path.append(".")
from modeling.unet import UNet
from data.transforms.transforms import setup_transform

# bad!
model = UNet()
model_path = "./output/1e-4batch2_bce0.5_ERGN/model_final.pt"
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
_, val_transform = setup_transform((384, 160))


def predict(image, mask):
    transformed = val_transform(image=image, mask=mask)
    image = transformed["image"].unsqueeze(0)
    mask = transformed["mask"]
    pred = F.sigmoid(model(image)).squeeze().detach().numpy()
    pred = ((pred > 0.5) * 255).astype(np.uint8)

    image = image.squeeze().numpy()
    _, axes = plt.subplots(1, 3, tight_layout=True)
    for i in range(3):
        axes[i].axis("off")
    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")

    plot_location(axes[1], image, pred)
    plot_segmentation(axes[2], image, pred)

    pred_labels, pred_num = measure.label(pred, return_num=True)
    mask_labels, mask_num = measure.label(mask, return_num=True)

    dice_coefficients = []
    for i in range(1, mask_num + 1):
        dc = 0
        for j in range(1, pred_num + 1):
            dc = max(dc, calculate_dice_cofficient(pred_labels, mask_labels, i, j))
        dice_coefficients.append(dc)

    dcs_formatted = [f"{dc:.2f}:" for dc in dice_coefficients]
    dc_formatted = f"{np.average(dice_coefficients):.2f}"

    plt.savefig("./result.png")
    result = imread("./result.png")
    os.remove("./result.png")

    return result, mask_num, pred_num, dcs_formatted, dc_formatted


def plot_location(interface, image, pred):
    interface.imshow(image, cmap="gray")
    # label connected regions of an integer array (1, 2, 3...)
    # default: 1(8)-connectivity, background = 0
    pred_labels = measure.label(pred)
    regions = measure.regionprops(pred_labels)
    x = [int(r.centroid[0]) for r in regions]
    y = [int(r.centroid[1]) for r in regions]
    interface.scatter(y, x, s=2, c="r")
    interface.axis("off")


def plot_segmentation(interface, image, pred):
    # sobel + watershed, gray2rgb is called, result become darker?
    # interface.imshow(mark_boundaries(image, pred, mode="inner", color=(1, 0, 0)))
    boundary = find_boundaries(pred, connectivity=2)
    interface.imshow(image, cmap="gray")
    interface.imshow(
        boundary, cmap="rainbow", alpha=0.5 * boundary, interpolation="nearest"
    )
    interface.axis("off")


def plot_location(interface, image, pred):
    interface.imshow(image, cmap="gray")
    # label connected regions of an integer array (1, 2, 3...)
    # default: 1(8)-connectivity, background = 0
    pred_labels = measure.label(pred)
    regions = measure.regionprops(pred_labels)
    x = [int(r.centroid[0]) for r in regions]
    y = [int(r.centroid[1]) for r in regions]
    interface.scatter(y, x, s=2, c="r")
    interface.axis("off")


def plot_segmentation(interface, image, pred):
    # sobel + watershed, gray2rgb is called, result become darker?
    # interface.imshow(mark_boundaries(image, pred, mode="inner", color=(1, 0, 0)))
    boundary = find_boundaries(pred, connectivity=2)
    interface.imshow(image, cmap="gray")
    interface.imshow(
        boundary, cmap="rainbow", alpha=0.5 * boundary, interpolation="nearest"
    )
    interface.axis("off")


def calculate_dice_cofficient(pred_labels, mask_labels, i, j):
    mask_map = mask_labels == i
    pred_map = pred_labels == j

    smooth = 1.0
    intersection = (mask_map * pred_map).sum()
    return (2.0 * intersection + smooth) / (mask_map.sum() + pred_map.sum() + smooth)


def main():
    demo = gr.Interface(
        title="Vertebra Localization and Segmentation",
        fn=predict,
        # inputs=gr.Image(image_mode="L", height=500, width=800),
        inputs=[
            gr.Image(label="Input", image_mode="L", height=400),
            gr.Image(label="Ground truth", image_mode="L", height=400),
        ],
        outputs=[
            gr.Image(height=480),
            gr.Textbox(label="# of vertebra (GT)"),
            gr.Textbox(label="# of vertebra (Detected)"),
            gr.Textbox(label="Dice coefficient of each vertebra"),
            gr.Textbox(label="Average Dice coefficient"),
        ],
        allow_flagging="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
