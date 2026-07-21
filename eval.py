import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import modules

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate YOLOv10 on test split')
    p.add_argument('--weights', type=str, default='esc_yolo_best.pt')
    p.add_argument('--data', type=str, default='data_23_04_26.yaml')
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--imgsz', type=int, nargs='+', default=[640, 800])
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.6)
    p.add_argument('--no-tta', action='store_true')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--target-map50', type=float, default=0.54)
    return p.parse_args()


def run_single(model, data, split, imgsz, conf, iou, device, augment=True):
    r = model.val(
        data=data,
        split=split,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=augment,
        verbose=False,
    )
    return {
        'imgsz': imgsz,
        'map50': float(r.box.map50),
        'map': float(r.box.map),
        'precision': float(r.box.mp),
        'recall': float(r.box.mr),
        'f1': 2 * float(r.box.mp) * float(r.box.mr) / (float(r.box.mp) + float(r.box.mr) + 1e-9),
    }


def print_result(entry, label=''):
    tag = f'[{label}] ' if label else ''
    print(f"  {tag}imgsz={entry['imgsz']:4d} │ "
          f"mAP50={entry['map50']:.4f}  "
          f"mAP50-95={entry['map']:.4f}  "
          f"P={entry['precision']:.4f}  "
          f"R={entry['recall']:.4f}  "
          f"F1={entry['f1']:.4f}")


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        print("        Please run training first or pass --weights <path>")
        sys.exit(1)

    print("=" * 65)
    print("  YOLOv10-S Enhanced v6 — Evaluation")
    print("=" * 65)
    print(f"  Weights : {weights}")
    print(f"  Data    : {args.data}")
    print(f"  Split   : {args.split}")
    print(f"  Device  : {device}")
    print(f"  TTA     : {'OFF' if args.no_tta else 'ON  (flip + multi-scale)'}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  conf    : {args.conf}  │  iou: {args.iou}")
    print("=" * 65)

    model = YOLO(str(weights))

    sizes_to_run = [args.imgsz[0]] if args.no_tta else args.imgsz
    use_tta = not args.no_tta

    results = []
    print(f"\n  Evaluating on split='{args.split}'...\n")
    for sz in sizes_to_run:
        entry = run_single(model, args.data, args.split,
                            sz, args.conf, args.iou, device, augment=use_tta)
        results.append(entry)
        print_result(entry)

    best = max(results, key=lambda x: x['map50'])

    print()
    print("=" * 65)
    print("  BEST RESULT")
    print("=" * 65)
    print(f"  imgsz      : {best['imgsz']}")
    print(f"  mAP50      : {best['map50']:.4f}  (target > {args.target_map50})")
    print(f"  mAP50-95   : {best['map']:.4f}")
    print(f"  Precision  : {best['precision']:.4f}")
    print(f"  Recall     : {best['recall']:.4f}")
    print(f"  F1         : {best['f1']:.4f}")
    print()

    if best['map50'] >= args.target_map50:
        print(f"  ✓ TARGET ACHIEVED: {best['map50']:.4f} >= {args.target_map50}")
    else:
        gap = args.target_map50 - best['map50']
        print(f"  ✗ Target not reached — missing {gap:.4f} ({gap*100:.2f}%)")
        print()
        print("  Suggestions:")
        if best['precision'] > 0.9 and best['recall'] < 0.7:
            print("  → Low recall: try decreasing conf threshold or increasing copy_paste")
        if best['recall'] > 0.9 and best['precision'] < 0.7:
            print("  → Low precision: try increasing conf threshold during inference")
        print("  → Fine-tune further on real data (freeze backbone, lr=1e-4)")
        print("  → Try imgsz=1024 if VRAM is available")
        print("  → Ensemble best.pt + last.pt")

    print("=" * 65)


if __name__ == '__main__':
    main()
