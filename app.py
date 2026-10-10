import asyncio
import glob
import hashlib
import io
import json
import os
import platform
import random
import shlex
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import clip
import numpy as np
import pillow_heif
import rawpy
import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

default_folder_name = Path("default_folder_name.txt").read_text().strip()


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

pt_state = torch.load(state_name, map_location=device)
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Serialize all ML inference through a single lock to avoid CUDA contention
ml_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

is_running = False
scores_cache: dict[str, float] = {}
all_processed_images: list[tuple[str, float]] = []
history_index = -1
images_in_folder: list[str] = []
images_in_folder_index: dict[str, int] = {}
current_showing_index = -1
last_processed_folder = ""

eval_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval")

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def get_image_features(image):
    img_t = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(img_t)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().detach().numpy()


def rotate_by_exif(image):
    if not hasattr(image, "_getexif"):
        return image
    try:
        exif = image._getexif()
        if exif:
            orientation = exif.get(0x0112)
            if orientation == 3:
                image = image.rotate(180)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def process_raw_image(image_path):
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8)
    return Image.fromarray(rgb)


def get_image(image_path: str) -> Image.Image:
    if image_path.lower().endswith(".heic"):
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride
        )
    elif image_path.lower().endswith((".arw", ".nef")):
        jpg_path = os.path.splitext(image_path)[0] + ".JPG"
        if os.path.exists(jpg_path):
            image = Image.open(jpg_path)
        else:
            image = process_raw_image(image_path)
    else:
        image = Image.open(image_path)
    image = rotate_by_exif(image)
    image.thumbnail((1920, 1920), Image.LANCZOS)
    return image


def get_score(image: Image.Image) -> float:
    with ml_lock:
        features = get_image_features(image)
        score = predictor(torch.from_numpy(features).to(device).float())
    return score.item()


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_files_in_folder(folder_path: str) -> list[str]:
    extensions = (".jpg", ".jpeg", ".arw", ".nef", ".heic")
    pattern = os.path.join(folder_path, "**", "*.*")
    return [f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(extensions)]


def update_stack(folder_path: str) -> list[str]:
    global scores_cache, all_processed_images, history_index
    global images_in_folder, images_in_folder_index, last_processed_folder

    if folder_path != last_processed_folder or not images_in_folder:
        files = find_files_in_folder(folder_path)
        files.sort()
        images_in_folder[:] = files
        images_in_folder_index.clear()
        images_in_folder_index.update({p: i for i, p in enumerate(files)})
        scores_cache.clear()
        all_processed_images.clear()
        history_index = -1
        last_processed_folder = folder_path
        print(f"{len(images_in_folder)} images in {folder_path}")

    sample = list(images_in_folder)
    random.shuffle(sample)
    return sample[:100]


# ---------------------------------------------------------------------------
# Evaluation generator
# ---------------------------------------------------------------------------


def process_images(folder_path: str, threshold: float):
    global is_running, current_showing_index, history_index

    current_images = update_stack(folder_path)
    total = len(current_images)

    for i, image_path in enumerate(current_images):
        if not is_running:
            break
        try:
            image = get_image(image_path)
            score = get_score(image)
            scores_cache[image_path] = score
            all_processed_images.append((image_path, score))
            history_index = len(all_processed_images) - 1
            current_showing_index = images_in_folder_index.get(image_path, -1)
            print(f"[{i + 1}/{total}] {os.path.basename(image_path)}: {score:.2f}")

            yield {"type": "update", "path": image_path, "score": score, "index": i + 1, "total": total}

            if score >= threshold:
                is_running = False
                break
        except Exception as e:
            print(f"Error {image_path}: {e}")

    yield {"type": "done"}


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def _cached_score(path: str) -> float:
    if path in scores_cache:
        return scores_cache[path]
    score = get_score(get_image(path))
    scores_cache[path] = score
    return score


def nav_prev_folder():
    global current_showing_index, is_running
    is_running = False
    if not images_in_folder:
        return {"error": "画像が読み込まれていません"}
    if current_showing_index > 0:
        current_showing_index -= 1
        path = images_in_folder[current_showing_index]
        score = _cached_score(path)
        return {"path": path, "score": score, "status": f"前の写真 ({current_showing_index + 1}/{len(images_in_folder)})"}
    return {"error": "これ以上前の画像はありません"}


def nav_next_folder():
    global current_showing_index, is_running
    is_running = False
    if not images_in_folder:
        return {"error": "画像が読み込まれていません"}
    if current_showing_index < len(images_in_folder) - 1:
        current_showing_index += 1
        path = images_in_folder[current_showing_index]
        score = _cached_score(path)
        return {"path": path, "score": score, "status": f"次の写真 ({current_showing_index + 1}/{len(images_in_folder)})"}
    return {"error": "これ以上次の画像はありません"}


def nav_prev_history():
    global history_index, current_showing_index, is_running
    is_running = False
    if not all_processed_images:
        return {"error": "評価済み画像がありません"}
    if history_index > 0:
        history_index -= 1
        path, score = all_processed_images[history_index]
        current_showing_index = images_in_folder_index.get(path, current_showing_index)
        return {"path": path, "score": score, "status": f"履歴 ({history_index + 1}/{len(all_processed_images)})"}
    return {"error": "履歴の先頭です"}


def nav_next_history():
    global history_index, current_showing_index, is_running
    is_running = False
    if not all_processed_images:
        return {"error": "評価済み画像がありません"}
    if history_index < len(all_processed_images) - 1:
        history_index += 1
        path, score = all_processed_images[history_index]
        current_showing_index = images_in_folder_index.get(path, current_showing_index)
        return {"path": path, "score": score, "status": f"履歴 ({history_index + 1}/{len(all_processed_images)})"}
    return {"error": "履歴の末尾です"}


NAV_FUNCS = {
    "prev_folder": nav_prev_folder,
    "next_folder": nav_next_folder,
    "prev_history": nav_prev_history,
    "next_history": nav_next_history,
}

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def copy_image(image_path: str) -> str:
    if not image_path:
        return "画像がありません"
    good_folder = "good"
    os.makedirs(good_folder, exist_ok=True)

    original_path = image_path
    for suffix in (".NEF", ".ARW"):
        raw_path = os.path.splitext(image_path)[0] + suffix
        if os.path.exists(raw_path):
            original_path = raw_path
            break

    dest_path = os.path.join(good_folder, os.path.basename(original_path))

    def file_hash(fp):
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(dest_path):
        if file_hash(original_path) == file_hash(dest_path):
            return f"既に保存済み: {dest_path}"
        base, ext = os.path.splitext(dest_path)
        i = 2
        while os.path.exists(f"{base}-{i}{ext}"):
            i += 1
        dest_path = f"{base}-{i}{ext}"

    shutil.copy(original_path, dest_path)
    return f"保存しました: {dest_path}"


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI()

_static_dir = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/default-folder")
async def get_default_folder():
    return {"folder": default_folder_name}


@app.get("/api/browse")
async def browse():
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$d.Description = 'フォルダを選択';"
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    folder = result.stdout.strip()
    return {"folder": folder if folder else None}


@app.get("/api/image")
async def serve_image(path: str):
    try:
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, get_image, path)
        data = await loop.run_in_executor(None, image_to_jpeg_bytes, img)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/evaluate")
async def evaluate(request: Request, folder: str = "", threshold: float = 5.0):
    global is_running

    async def event_gen():
        global is_running
        is_running = True
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for result in process_images(folder, threshold):
                    asyncio.run_coroutine_threadsafe(q.put(result), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(q.put({"type": "error", "message": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        fut = loop.run_in_executor(eval_executor, worker)

        try:
            while True:
                if await request.is_disconnected():
                    is_running = False
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            is_running = False
            await fut

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stop")
async def stop_eval():
    global is_running
    is_running = False
    return {"message": "停止しました"}


@app.get("/api/navigate")
async def navigate(direction: str):
    global is_running
    is_running = False  # stop any running evaluation first
    func = NAV_FUNCS.get(direction)
    if not func:
        return JSONResponse({"error": "Invalid direction"}, status_code=400)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, func)
    return result


@app.post("/api/save")
async def save(request: Request):
    data = await request.json()
    path = data.get("path", "")
    loop = asyncio.get_running_loop()
    message = await loop.run_in_executor(None, copy_image, path)
    return {"message": message}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
