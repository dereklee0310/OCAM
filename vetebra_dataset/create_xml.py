import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch.utils.data


def find_contour(mask_img, thickness=-1):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    separated_instances = np.zeros_like(mask_img)
    color_give = 255
    for i, contour in enumerate(contours):
        cv2.drawContours(separated_instances, [contour], -1, color=color_give - i, thickness=thickness)

    return separated_instances


def create_xml_annotation(image_path, image_name, image_size, objects, output_dir):
    """
    Creates and saves an XML annotation file for a given image and its objects.

    Args:
        image_name (str): The name of the image.
        image_size (tuple): The size of the image as (width, height, depth).
        objects (list): List of dictionaries containing object properties.
        output_dir (str): Directory where the XML file will be saved.
    """
    annotation = ET.Element("annotation", verified="no")
    ET.SubElement(annotation, "folder").text = "label"
    ET.SubElement(annotation, "filename").text = str(image_name)
    ET.SubElement(annotation, "path").text = image_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_size[1])
    ET.SubElement(size, "height").text = str(image_size[0])
    ET.SubElement(size, "depth").text = str(1)
    ET.SubElement(annotation, "segmented").text = "0"

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "type").text = "robndbox"
        ET.SubElement(obj_elem, "name").text = obj["name"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        robndbox = ET.SubElement(obj_elem, "robndbox")
        ET.SubElement(robndbox, "cx").text = str(obj["cx"])
        ET.SubElement(robndbox, "cy").text = str(obj["cy"])
        ET.SubElement(robndbox, "w").text = str(obj["width"])
        ET.SubElement(robndbox, "h").text = str(obj["height"])
        ET.SubElement(robndbox, "angle").text = str(obj["angle"])

    xml_str = ET.tostring(annotation, encoding="unicode")
    xml_file_path = os.path.join(output_dir, "00"+str(image_name) + ".xml")
    print(xml_file_path)
    with open(xml_file_path, "w") as xml_file:
        xml_file.write(xml_str)
        
    if os.path.exists(xml_file_path):
        return True
    else:
        return False


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, output_dir="./xml_output"):
        self.root = root
        self.transforms = transforms
        self.output_dir = output_dir
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "label"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        print(img_path)
        filename = os.path.basename(img_path)
        file = int(os.path.splitext(filename)[0])
        mask_path = os.path.join(self.root, "label", filename)
        print(mask_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = find_contour(mask)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        print("num_objs:", num_objs)
        objects = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            cx = (xmax+xmin)/2.0
            cy = (ymax+ymin)/2.0
            w = xmax-xmin
            h = ymax-ymin
            objects.append({"name": f"v{i}", "cx": cx, "cy": cy, "width": w, "height": h, "angle": 0.0})
        
        print(objects)
        print("create")
        create_xml_annotation(img_path, file, img.shape[:3], objects, self.output_dir)

        if self.transforms:
            img = self.transforms(img)
        return  img, object

    def __len__(self):
        return len(self.imgs)


def main():
    dataset = CustomDataset(root='./f03')# root=path/to/image & label mask
    for i in range(len(dataset)):
        img, object = dataset[i]

if __name__ == '__main__':
    main()
