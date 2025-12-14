import os, argparse, torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing train/val/test and *_coco.json files")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=100, help="Sanity steps (set 0 to use full epoch)")
    ap.add_argument("--out_dir", default="outputs/model")
    return ap.parse_args()

class CocoDetrDataset(Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2idx = {cid: i for i, cid in enumerate(self.cat_ids)}  # to 0..K-1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        clean_anns = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            clean_anns.append({
                "bbox": [float(x), float(y), float(w), float(h)],
                "category_id": int(self.cat2idx[a["category_id"]]),
                "area": float(a.get("area", w*h)),
                "iscrowd": int(a.get("iscrowd", 0)),
            })

        target = {"image_id": int(img_id), "annotations": clean_anns}
        return image, target

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img = os.path.join(args.data_dir, "train")
    train_ann = os.path.join(args.data_dir, "train_coco.json")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def collate_fn(batch):
        images, targets = zip(*batch)
        return processor(images=list(images), annotations=list(targets), return_tensors="pt")

    train_ds = CocoDetrDataset(train_img, train_ann)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)

    num_classes = len(train_ds.cat_ids)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    steps = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = dict(batch)
            batch["pixel_values"] = batch["pixel_values"].to(device)
            if "pixel_mask" in batch:
                batch["pixel_mask"] = batch["pixel_mask"].to(device)

            # move nested label tensors
            new_labels = []
            for t in batch["labels"]:
                t = dict(t)
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(device)
                new_labels.append(t)
            batch["labels"] = new_labels

            out = model(**batch)
            loss = out.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            steps += 1
            if steps % 20 == 0:
                print(f"epoch {epoch+1} step {steps} loss {float(loss):.4f}")

            if args.max_steps > 0 and steps >= args.max_steps:
                break

        if args.max_steps > 0 and steps >= args.max_steps:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print("Saved model to:", args.out_dir)
    print("CUDA:", torch.cuda.is_available(), "num_classes:", num_classes)

if __name__ == "__main__":
    main()
