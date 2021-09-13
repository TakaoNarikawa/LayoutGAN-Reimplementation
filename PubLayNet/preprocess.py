import json
from itertools import groupby

import numpy as np

# category = {
#     1: "text",
#     2: "title",
#     3: "list",
#     4: "table",
#     5: "figure"
# }

NUM = 9

def process(json_path, output_name):

    print("Process start:", json_path, output_name)
    
    with open(json_path) as f:
        data = json.load(f)
    
    annotations = [
        {
            "image_id": a["image_id"],
            "bbox": a["bbox"],
            "category_id": a["category_id"]
        }
        for a in data['annotations']
    ]
    images = {
        a["id"]: (a["width"], a["height"])
        for a in data['images']
    }

    def handle_batch(image_id, batch):
        W, H = images[image_id]
        bboxes = np.zeros((9, 4 + 5))
        for i, b in enumerate(batch):
            x, y, w, h = b["bbox"]
            xc = x + w / 2
            yc = y + h / 2
            label_index = (b["category_id"] - 1) + 4
            bboxes[i, :4] = [xc / W, yc / H, w / W, h / H]
            bboxes[i, label_index] = 1
        return bboxes

    output_data = [
        (image_id, list(batch))
        for image_id, batch in groupby(annotations, key=lambda a: a['image_id'])
    ]
    output_data = np.array([
        handle_batch(image_id, batch)
        for image_id, batch in output_data
        if 0 < len(batch) <= NUM
    ])

    np.save(output_name, output_data)


if __name__ == "__main__":
    process("val.json", "val")
    process("train.json", "train")
