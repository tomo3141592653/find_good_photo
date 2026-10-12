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
LAST_FOLDER_FILE = Path(__file__).parent / "last_folder.txt"


def get_last_folder() -> str:
    try:
        folder = LAST_FOLDER_FILE.read_text(encoding="utf-8").strip()
        if folder and os.path.isdir(folder):
            return folder
    except OSError:
        pass
    return default_folder_name


def remember_folder(folder: str):
    try:
        LAST_FOLDER_FILE.write_text(folder, encoding="utf-8")
    except OSError:
        pass


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

# Each entry is the list of files created by one save action (single or bulk)
save_log: list[list[str]] = []

CACHE_DIR = Path(__file__).parent / "score_cache"

# ---------------------------------------------------------------------------
# Persistent score cache (per folder, keyed by path + mtime + size)
# ---------------------------------------------------------------------------


def _cache_file(folder_path: str) -> Path:
    key = hashlib.md5(os.path.normcase(os.path.abspath(folder_path)).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def _file_sig(path: str) -> Optional[list]:
    try:
        st = os.stat(path)
        return [st.st_mtime, st.st_size]
    except OSError:
        return None


def load_score_cache(folder_path: str, files: list[str]) -> dict[str, float]:
    cf = _cache_file(folder_path)
    if not cf.exists():
        return {}
    try:
        data = json.loads(cf.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if data.get("model") != state_name:
        return {}
    saved = data.get("scores", {})
    loaded = {}
    for p in files:
        entry = saved.get(p)
        if entry and _file_sig(p) == entry[1:]:
            loaded[p] = entry[0]
    return loaded


def save_score_cache(folder_path: str):
    scores = {}
    for p, s in list(scores_cache.items()):
        sig = _file_sig(p)
        if sig:
            scores[p] = [s, *sig]
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _cache_file(folder_path).write_text(
            json.dumps({"model": state_name, "scores": scores}, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"Cache save failed: {e}")

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
        scores_cache.update(load_score_cache(folder_path, files))
        all_processed_images.clear()
        history_index = -1
        last_processed_folder = folder_path
        print(f"{len(images_in_folder)} images in {folder_path} ({len(scores_cache)} cached scores)")

    sample = list(images_in_folder)
    random.shuffle(sample)
    return sample[:100]


# ---------------------------------------------------------------------------
# Evaluation generator
# ---------------------------------------------------------------------------


def process_images(folder_path: str, threshold: float):
    current_images = update_stack(folder_path)
    try:
        yield from _process_loop(current_images, threshold, folder_path)
    finally:
        save_score_cache(folder_path)


def _process_loop(current_images, threshold, folder_path):
    global is_running, current_showing_index, history_index
    total = len(current_images)
    new_count = 0

    for i, image_path in enumerate(current_images):
        if not is_running:
            break
        try:
            score = scores_cache.get(image_path)
            if score is None:
                score = get_score(get_image(image_path))
                scores_cache[image_path] = score
                new_count += 1
                if new_count % 20 == 0:
                    save_score_cache(folder_path)
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


def copy_image(image_path: str, created: Optional[list] = None) -> str:
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
    if created is not None:
        created.append(dest_path)
    return f"保存しました: {dest_path}"


def save_images(paths: list[str]) -> str:
    if not paths:
        return "画像がありません"
    created: list[str] = []
    messages = [copy_image(p, created) for p in paths]
    if created:
        save_log.append(created)
    if len(paths) == 1:
        return messages[0]
    skipped = len(paths) - len(created)
    return f"{len(created)} 枚を保存しました" + (f"（{skipped} 枚は保存済み/失敗）" if skipped else "")


def undo_last_save() -> str:
    if not save_log:
        return "取り消せる保存がありません"
    created = save_log.pop()
    removed = 0
    for dest in created:
        try:
            os.remove(dest)
            removed += 1
        except OSError:
            pass
    return f"保存を取り消しました（{removed} 枚削除）"


LIGHTROOM_CANDIDATES = [
    r"C:\Program Files\Adobe\Adobe Lightroom Classic\Lightroom.exe",
    r"C:\Program Files\Adobe\Adobe Lightroom CC\lightroom.exe",
]


def find_original(image_path: str) -> str:
    for suffix in (".NEF", ".ARW"):
        raw_path = os.path.splitext(image_path)[0] + suffix
        if os.path.exists(raw_path):
            return raw_path
    return image_path


def open_in_lightroom(path: str) -> str:
    if not path or not os.path.exists(path):
        return "ファイルが見つかりません"
    exe = next((c for c in LIGHTROOM_CANDIDATES if os.path.exists(c)), None)
    if not exe:
        return "Lightroom が見つかりません"
    target = find_original(path)
    subprocess.Popen([exe, os.path.normpath(target)])
    return f"Lightroom で開きます: {os.path.basename(target)}"


def reveal_in_explorer(path: str) -> str:
    if not path or not os.path.exists(path):
        return "ファイルが見つかりません"
    if platform.system() == "Windows":
        subprocess.Popen(["explorer", f"/select,{os.path.normpath(path)}"])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])
    return "フォルダを開きました"


def open_saved_folder() -> str:
    # copy_image と同じく作業フォルダ基準の good を開く
    folder = os.path.abspath("good")
    os.makedirs(folder, exist_ok=True)
    if platform.system() == "Windows":
        subprocess.Popen(["explorer", os.path.normpath(folder)])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])
    return "保存先のフォルダを開きました"


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
    return {"folder": get_last_folder()}


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
    if folder:
        remember_folder(folder)
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

    if folder and os.path.isdir(folder):
        remember_folder(folder)

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
    message = await loop.run_in_executor(None, save_images, [path] if path else [])
    return {"message": message}


@app.post("/api/undo-save")
async def undo_save():
    loop = asyncio.get_running_loop()
    return {"message": await loop.run_in_executor(None, undo_last_save)}


@app.post("/api/lightroom")
async def lightroom(request: Request):
    data = await request.json()
    return {"message": open_in_lightroom(data.get("path", ""))}


@app.post("/api/reveal")
async def reveal(request: Request):
    data = await request.json()
    return {"message": reveal_in_explorer(data.get("path", ""))}


@app.post("/api/open-saved")
async def open_saved():
    return {"message": open_saved_folder()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
