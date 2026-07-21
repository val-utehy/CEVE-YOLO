import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modules

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Run YOLOv10 inference')
    p.add_argument('--weights', type=str, default='esc_yolo_best.pt')
    p.add_argument('--source', type=str, required=True, help='Path to image, folder, video, or glob pattern')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--save-dir', type=str, default='runs/predict')
    p.add_argument('--save-txt', action='store_true')
    p.add_argument('--save-conf', action='store_true')
    p.add_argument('--no-save-img', action='store_true')
    p.add_argument('--show', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    print("=" * 65)
    print("  YOLOv10-S — Inference")
    print("=" * 65)
    print(f"  Weights : {weights}")
    print(f"  Source  : {source}")
    print(f"  Device  : {device}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  conf    : {args.conf}  │  iou: {args.iou}")
    print("=" * 65)

    model = YOLO(str(weights))

    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=not args.no_save_img,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=str(Path(args.save_dir).parent) if Path(args.save_dir).parent != Path('.') else 'runs',
        name=Path(args.save_dir).name,
        exist_ok=True,
        verbose=False,
    )

    total_dets = sum(len(r.boxes) for r in results)
    print(f"\n  Processed {len(results)} image(s), {total_dets} detection(s) total")
    if not args.no_save_img:
        print(f"  Results saved to: {results[0].save_dir}")
    print("=" * 65)


if __name__ == '__main__':
    main()
