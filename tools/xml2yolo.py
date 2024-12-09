import json
import os
import shutil
import xml.etree.ElementTree as ET
import math
import cv2 as cv
from PIL import Image


def voc_to_dota(result_path, xml_path, total_label):

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_, _name = os.path.split(xml_path)
    temp_name, ext = os.path.splitext(_name)
    tree = ET.parse(os.path.join(xml_path))
    root = tree.getroot()
    # print(root[6][0].text)
    filename = temp_name + ".jpg"
    txt_name = temp_name + ".txt"
    # img = cv.imread(os.path.join(file_,filename))
    txt_file = os.path.join(result_path, txt_name)

    with open(txt_file, "w+", encoding="UTF-8") as out_file:
        # out_file.write('imagesource:null' + '\n' + 'gsd:null' + '\n')
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in total_label:
                total_label.append(name)
            difficult = obj.find("difficult").text
            # print(name, difficult)
            robndbox = obj.find("robndbox")
            cx = float(robndbox.find("cx").text)
            cy = float(robndbox.find("cy").text)
            w = float(robndbox.find("w").text)
            h = float(robndbox.find("h").text)
            angle = float(robndbox.find("angle").text)
            # print(cx, cy, w, h, angle)
            p0x, p0y = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
            p1x, p1y = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
            p2x, p2y = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
            p3x, p3y = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

            dict = {p0y: p0x, p1y: p1x, p2y: p2x, p3y: p3x}
            list = find_topLeftPopint(dict)
            # print((list))
            if list[0] == p0x:
                list_xy = [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]
            elif list[0] == p1x:
                list_xy = [p1x, p1y, p2x, p2y, p3x, p3y, p0x, p0y]
            elif list[0] == p2x:
                list_xy = [p2x, p2y, p3x, p3y, p0x, p0y, p1x, p1y]
            else:
                list_xy = [p3x, p3y, p0x, p0y, p1x, p1y, p2x, p2y]

            # cv.line(img, (int(list_xy[0]), int(list_xy[1])), (int(list_xy[2]), int(list_xy[3])), color=(255, 0, 0), thickness= 3)
            # cv.line(img, (int(list_xy[2]), int(list_xy[3])), (int(list_xy[4]), int(list_xy[5])), color=(0, 255, 0), thickness= 3)
            # cv.line(img, (int(list_xy[4]), int(list_xy[5])), (int(list_xy[6]), int(list_xy[7])), color=(0, 0, 255), thickness= 2)
            # cv.line(img, (int(list_xy[6]), int(list_xy[7])), (int(list_xy[0]), int(list_xy[1])), color=(255, 255, 0), thickness= 2)

            # data = str(list_xy[0]) + " " + str(list_xy[1]) + " " + str(list_xy[2]) + " " + str(list_xy[3]) + " " + \
            # 	   str(list_xy[4]) + " " + str(list_xy[5]) + " " + str(list_xy[6]) + " " + str(list_xy[7]) + " "
            # data = data + name + " " + difficult + "\n"

            list_xy = [
                z / 500 if i % 2 == 0 else z / 1200 for i, z in enumerate(list_xy)
            ]
            data = "0 "  # class label
            data += (
                str(list_xy[0])
                + " "
                + str(list_xy[1])
                + " "
                + str(list_xy[2])
                + " "
                + str(list_xy[3])
                + " "
                + str(list_xy[4])
                + " "
                + str(list_xy[5])
                + " "
                + str(list_xy[6])
                + " "
                + str(list_xy[7])
                + " "
            )
            data += "\n"
            out_file.write(data)
            # cv.imwrite(os.path.join(result_path,filename),img)


def rolabelimg2lablelme(jpg_xml_path, verify_dir, text_path):
    if not os.path.exists(verify_dir):
        os.makedirs(verify_dir)
    txt_list = glob.glob(os.path.join(text_path, ".".join(["*", "txt"])))
    for i in range(len(txt_list)):
        (filepath, tempfilename) = os.path.split(txt_list[i])
        (filename, extension) = os.path.splitext(tempfilename)
        sourcePath = None
        image_filename = None
        if os.path.exists(os.path.join(jpg_xml_path, ".".join([filename, "jpg"]))):
            sourcePath = os.path.join(jpg_xml_path, ".".join([filename, "jpg"]))
            image_filename = ".".join([filename, "jpg"])
        elif os.path.exists(os.path.join(jpg_xml_path, ".".join([filename, "png"]))):
            sourcePath = os.path.join(jpg_xml_path, ".".join([filename, "png"]))
            image_filename = ".".join([filename, "png"])
        if sourcePath is None:
            print("check photo type")
            continue
        targetPath = verify_dir
        # shutil.copy(sourcePath, targetPath) # wtf ?
        img = Image.open(sourcePath)
        imgSize = img.size
        w = img.width
        h = img.height

        data = {}
        data["imagePath"] = image_filename
        data["flags"] = {}
        data["imageWidth"] = w
        data["imageHeight"] = h
        data["imageData"] = None
        data["version"] = "5.0.1"
        data["shapes"] = []

        with open(txt_list[i]) as f:
            label_str = f.readlines()
            for label_item in label_str:
                line_char = label_item.split("\n")[0].split(" ")
                points = [
                    [eval(line_char[0]), eval(line_char[1])],
                    [eval(line_char[2]), eval(line_char[3])],
                    [eval(line_char[4]), eval(line_char[5])],
                    [eval(line_char[6]), eval(line_char[7])],
                ]
                itemData = {"points": []}
                itemData["points"].extend(points)
                itemData["flag"] = {}
                itemData["group_id"] = None
                itemData["shape_type"] = "polygon"
                itemData["label"] = line_char[-2]
                data["shapes"].append(itemData)

            jsonName = ".".join([filename, "json"])
            jsonPath = os.path.join(targetPath, jsonName)
            with open(jsonPath, "w") as f:
                json.dump(data, f)
            print(jsonName)
            print("dota2labelme...")


def find_topLeftPopint(dict):
    dict_keys = sorted(dict.keys())
    temp = [dict[dict_keys[0]], dict[dict_keys[1]]]
    minx = min(temp)
    if minx == temp[0]:
        miny = dict_keys[0]
    else:
        miny = dict_keys[1]
    return [minx, miny]


def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = -sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)

    return float(format(xc + pResx, ".1f")), float(format(yc + pResy, ".1f"))
    # return xc + pResx, yc + pResy


if __name__ == "__main__":
    jpg_xml_path = "./labelled_data/f03/label"  # 路徑記得改
    text_path = "./labelled_data/f03/txt"
    labelme_dir = "./labelled_data/f03/Json"
    label_list = []
    import glob

    file_glob = glob.glob(jpg_xml_path + "/*.xml")
    for item in file_glob:
        voc_to_dota(text_path, item, label_list)
    print("------")

    # generate json files, comment out to avoid overwrite because it's based on dota
    # format generated by voc_to_dota, which is incorrect
    # rolabelimg2lablelme(jpg_xml_path,labelme_dir,text_path)
    # print(label_list)
