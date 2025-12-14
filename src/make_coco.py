import os
import json
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with .jpg and matching .json files")
    ap.add_argument("--output_json", required=True, help="Output COCO json path")
    return ap.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_json = args.output_json

    images = []
    annotations = []
    categories = {}
    ann_id = 1
    img_id = 1

    files = sorted(os.listdir(input_dir))

    for f in files:
        if not f.endswith(".json"):
            continue

        json_path = os.path.join(input_dir, f)
        img_name = f.replace(".json", ".jpg")

        with open(json_path, "r") as jf:
            data = json.load(jf)

        # Register image (width/height unknown → set 0)
        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": 0,
            "height": 0
        })

        frames = data.get("frames", [])
        if not frames:
            img_id += 1
            continue

        objects = frames[0].get("objects", [])

        for obj in objects:
            cat = obj.get("category")
            box = obj.get("box2d")

            if cat is None or box is None:
                continue

            if cat not in categories:
                categories[cat] = len(categories) + 1

            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 1 or h <= 1:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": categories[cat],
                "bbox": [x1, y1, w, h],  # COCO format (xywh)
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        if img_id % 500 == 0:
            print(f"Processed {img_id} images...")

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f)

    print("Saved COCO file:", output_json)
    print("Images:", len(images))
    print("Annotations:", len(annotations))
    print("Categories:", len(categories))

if __name__ == "__main__":
    main()
