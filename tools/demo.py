import gradio as gr
import numpy as np
from PIL import Image

from inference import *


def predict(image):
    try:
        with open(CONFIG, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file)
    except yaml.YAMLError as e:
        print(e)

    # quick override for test
    cfg["save"] = False
    cfg["model"] = "runs/l_mosaic0_v0_batch8_epoch200/fold01/train/weights/best.pt"
    cfg["source"] = image
    cfg["show_labels"] = False
    cfg["show_conf"] = False
    model = YOLO(cfg["model"])
    result = model.predict(**cfg)[0]

    xywhr = result.obb.xywhr.to("cpu").numpy().copy().tolist()
    # xywhr = [y for x, y in zip(result.obb.conf, xywhr) if x >= 0.8]
    xywhr.sort(key=lambda x: x[1])  # sort by y coord
    angles = [x[4] for x in xywhr]
    max_angle_diff, idx1, idx2 = get_max_angle_diff(angles, xywhr)
    angle1 = radian2angle(xywhr[idx1][4])
    angle2 = radian2angle(xywhr[idx2][4])
    angle1, angle2 = max(angle1, angle2), min(angle1, angle2)

    # prediction
    result.save("pred.png", conf=False, labels=False)
    prediction = Image.open("pred.png")

    # image with boxes and liness
    image=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    xyxyxyxy = result.obb.xyxyxyxy.to("cpu").numpy().copy().astype(np.int32).tolist()
    xyxyxyxy = sorted(xyxyxyxy, key=lambda x:x[0][1]) # sort by y of first (x, y)
    upper, lower = sorted([xyxyxyxy[idx1], xyxyxyxy[idx2]], key=lambda x: x[0][1])
    draw_box_and_line(image, upper, upper=True)
    draw_box_and_line(image, lower, upper=False)
    image_box_line = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_box_line = Image.fromarray(image)
    image_box_line.save("box_line.png")

    match max_angle_diff:
        case x if x > 40:
            result = "severe scoliosis"
        case x if x >= 20:
            result = "moderate scoliosis"
        case x if x >= 10:
            result = "mild scoliosis"
        case x:
            result = "spinal curve"

    return np.concatenate((prediction, image_box_line), axis=1), max_angle_diff, result, angle1, angle2

def clear_input():
    return None, "", "", ""

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# OCAM:Oriented-Bounding-Box Cobb Angle Measurement for Scoliosis Diagonosis")
        with gr.Row():
            input_img = gr.Image(height=600, width=250, type="pil")
            output_img = gr.Image(height=600, width=500)
        with gr.Row():
            clear = gr.Button("clear", min_width=355)
            submit = gr.Button("submit", min_width=360)
            cobb_angle = gr.Textbox(label="Cobb angle", min_width=100)
            result = gr.Textbox(label="Result", min_width=100)
            max = gr.Textbox(label="Max theata", min_width=100)
            min = gr.Textbox(label="Min theata", min_width=100)

        submit.click(fn=predict, inputs=[input_img], outputs=[output_img, cobb_angle, result, max, min])
        clear.click(fn=clear_input, inputs=[], outputs=[output_img, cobb_angle, max, min])
        demo.launch(share=True)


if __name__ == "__main__":
    main()
