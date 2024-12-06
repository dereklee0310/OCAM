import xml.etree.ElementTree as ET

# XML data as a string (can also read from a file)
xml_data = ET.parse(r"vetebra_dataset\f02\label\0021.xml")

# Parse the XML data
root = xml_data.getroot()

# Extract general annotation data
annotation_data = {
    "verified": root.attrib.get("verified"),
    "folder": root.find("folder").text,
    "filename": root.find("filename").text,
    "path": root.find("path").text,
    "size": {
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
        "depth": int(root.find("size/depth").text),
    },
    "segmented": int(root.find("segmented").text),
}

# Extract object data
objects = []
for obj in root.findall("object"):
    obj_data = {
        "type": obj.find("type").text,
        "name": obj.find("name").text,
        "pose": obj.find("pose").text,
        "truncated": int(obj.find("truncated").text),
        "difficult": int(obj.find("difficult").text),
        "robndbox": {
            "cx": float(obj.find("robndbox/cx").text),
            "cy": float(obj.find("robndbox/cy").text),
            "w": float(obj.find("robndbox/w").text),
            "h": float(obj.find("robndbox/h").text),
            "angle": float(obj.find("robndbox/angle").text),
        },
    }
    objects.append(obj_data)

# Combine everything into one data structure
parsed_data = {
    "annotation": annotation_data,
    "objects": objects,
}

# Output the parsed data
print(parsed_data)
