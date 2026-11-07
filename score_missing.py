"""写真の一覧で美しさの点数が無い写真に、このアプリと同じ物差し（CLIP ViT-L/14 + sac+logos+ava1-l14-linearMSE）で点数を付ける。

アユムの毎晩の更新（ayumu-lab/repos/local-tools/scripts/photos_nightly.sh）が WSL から呼ぶ。
写真ビューア（tomopiviewer）が点数を付けていたが、写真ビューアの一覧に載らない写真（アプリを開かなくなってから撮った分など）には
点数が付かず、写真ランキングに出てこなかった（2026-11-07 時点で24万枚）。

  .venv\\Scripts\\python.exe score_missing.py --todo D:\\App_data\\ayumu-aes\\todo.json --out D:\\App_data\\ayumu-aes\\scores.tsv --minutes 180

todo.json は [[写真ID, Windows のパス], ...]。結果は scores.tsv に「写真ID<TAB>点数」を1行ずつ足す（既にある ID は飛ばす。止めても続きから）。
読めなかった写真は「写真ID<TAB>」（点数なし）で残し、次から飛ばす。
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pillow_heif
import torch
import torch.nn as nn
from PIL import Image, ImageOps

import clip

pillow_heif.register_heif_opener()
HERE = Path(__file__).parent
RAW = (".arw", ".nef", ".cr2", ".dng", ".raf", ".orf", ".rw2")


class AestheticPredictor(nn.Module):   # app.py と同じ形
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


def load(path):
    """同じコマの JPG があれば RAW の代わりに使う。JPEG は縮めて読む（CLIP は 224 しか見ない）"""
    try:
        base, ext = os.path.splitext(path)
        if ext.lower() in RAW:
            for e in (".JPG", ".jpg", ".JPEG", ".jpeg"):
                if os.path.exists(base + e):
                    path = base + e
                    break
            else:
                import rawpy
                with rawpy.imread(path) as r:
                    return Image.fromarray(r.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8))
        im = Image.open(path)
        if im.format == "JPEG":
            im.draft("RGB", (640, 640))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((640, 640))
        return im
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    t_end = time.time() + a.minutes * 60
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {l.split("\t", 1)[0] for l in f if l.strip()}
    todo = [(i, p) for i, p in json.load(open(a.todo, encoding="utf-8")) if i not in done]
    print(f"これから {len(todo)} 枚（済み {len(done)}）", flush=True)
    if not todo:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = AestheticPredictor(768)
    pred.load_state_dict(torch.load(HERE / "sac+logos+ava1-l14-linearMSE.pth", map_location=device))
    pred.to(device).eval()
    model, prep = clip.load("ViT-L/14", device=device)
    n = 0; t0 = time.time()
    with ThreadPoolExecutor(6) as ex, open(a.out, "a", encoding="utf-8") as out:
        for k in range(0, len(todo), a.batch):
            if time.time() > t_end:
                print("時間切れ。続きは次の回", flush=True)
                break
            chunk = todo[k:k + a.batch]
            ims = list(ex.map(lambda r: load(r[1]), chunk))
            ok = [(r, im) for r, im in zip(chunk, ims) if im is not None]
            for r, im in zip(chunk, ims):
                if im is None:
                    out.write(f"{r[0]}\t\n")
            if ok:
                x = torch.stack([prep(im) for _, im in ok]).to(device)
                with torch.no_grad():
                    f = model.encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True)
                    s = pred(f.float()).squeeze(1).tolist()
                for (r, _), v in zip(ok, s):
                    out.write(f"{r[0]}\t{v:.4f}\n")
            out.flush()
            n += len(chunk)
            if n % (a.batch * 50) < a.batch:
                el = time.time() - t0
                print(f"{n}/{len(todo)} 枚 {n / el:.1f} 枚/秒", flush=True)
    print(f"終わり {n} 枚 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
