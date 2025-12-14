import os, argparse, random, torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with val/ and val_coco.json")
    ap.add_argument("--ckpt", required=True, help="Folder saved by train.py (outputs/model)")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out_img", default="outputs/inference_thresh_02.png")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DetrImageProcessor.from_pretrained(args.ckpt)
    model = DetrForObjectDetection.from_pretrained(args.ckpt).to(device)
    model.eval()

    val_ann = os.path.join(args.data_dir, "val_coco.json")
    val_dir = os.path.join(args.data_dir, "val")
    coco = COCO(val_ann)

    ids = list(coco.imgs.keys())
    img_id = random.choice(ids)
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(val_dir, info["file_name"])
    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (h, w)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=args.threshold)[0]

    label_names = [coco.cats[cid]["name"] for cid in sorted(coco.getCatIds())]

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label_names[int(label)]} {float(score):.2f}", color="white",
                bbox=dict(facecolor="red", alpha=0.6, pad=2))

    plt.axis("off")
    os.makedirs(os.path.dirname(args.out_img), exist_ok=True)
    plt.savefig(args.out_img, bbox_inches="tight", dpi=150)
    plt.show()

    print("Saved:", args.out_img)
    print("Predicted boxes:", len(results["boxes"]))

if __name__ == "__main__":
    main()
