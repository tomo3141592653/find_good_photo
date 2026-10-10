import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Generator, List, Tuple
import hashlib
import platform
import shlex
import numpy as np
from PIL import Image
import requests
import torch
import torch.nn as nn

import clip
import gradio as gr
import pillow_heif
import rawpy


default_folder_name = Path("default_folder_name.txt").read_text().strip()

# AestheticPredictorクラスの定義
# このクラスは、画像の美的評価を予測するためのニューラルネットワークモデルを表します

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
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


# Download and load the model
state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)

#state_name = "best_model_valid_loss_0.3307_epoch17.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
pt_state = torch.load(state_name, map_location=device)
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


# Global variables to keep track of the current state
current_showing_index = -1  # images_in_folder内のインデックス
is_running = False
scores_cache = {}  # {path: score}
all_processed_images = []   # [(path, score), ...] スコア計算順
history_index = -1          # all_processed_images内のインデックス
images_in_folder = []  # ファイル名順ソート済み
images_in_folder_index = {}  # {path: index} O(1)検索用
last_processed_folder = ""

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

def rotate_by_exif(image):
    if not hasattr(image, '_getexif'):
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
    except Exception as e:
        print(f"画像の回転中にエラーが発生しました: {str(e)}")
    return image

def process_raw_image(image_path):
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=True,
            no_auto_bright=True,
            output_bps=8,
        )
    return Image.fromarray(rgb)


def get_image(image_path:str):
    """
   
    指定された画像パスから画像を読み込む関数。

    Args:
        image_path (str): 画像ファイルへのパス。

    Returns:
        PIL.Image.Image: 読み込まれた画像オブジェクト。

    Note:
        - JPG、JPEG、の画像ファイルを直接開きます。
        - ARWまたはNEF形式（RAWファイル）の場合：
          - 同名のJPGファイルが存在すれば、それを優先して開きます。
          - JPGが見つからない場合、RAWファイルを直接処理します。
    """

    if image_path.lower().endswith('.heic'):
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

    elif image_path.lower().endswith(('.arw', '.nef')):
        # JPG版を探す
        jpg_path = os.path.splitext(image_path)[0] + '.JPG'
        if os.path.exists(jpg_path):
            image = Image.open(jpg_path)
        else:
            image = process_raw_image(image_path)
    else:
        image = Image.open(image_path)

    image = rotate_by_exif(image)
    image.thumbnail((1000,1000),Image.LANCZOS)
    return image

def get_score(image):

    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

def find_files_in_folder(folder_path):
    extensions = ('.jpg', '.jpeg', '.arw', '.nef', '.heic')
    
    if platform.system() == 'Windows':
        # Windowsの場合
        import glob
        pattern = os.path.join(folder_path, '**', '*.*')
        files = [f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(extensions)]
    else:
        # Unix系OSの場合
        extensions_regex = r'\.jpe?g$|\.arw$|\.nef$|\.heic$'
        escaped_folder_path = shlex.quote(folder_path)
        command = f"find {escaped_folder_path} -type f | grep -E -i '({extensions_regex})'"
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            files = result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"エラーが発生しました: {e}")
            files = []
    
    return files
    
def update_stack(folder_path: str) -> List[str]:
    global current_showing_index, is_running, scores_cache, all_processed_images, history_index, images_in_folder, images_in_folder_index, last_processed_folder

    if folder_path != last_processed_folder or not images_in_folder:
        files = find_files_in_folder(folder_path)
        files.sort()  # ファイル名順 = 撮影時刻順
        images_in_folder = files
        images_in_folder_index = {path: i for i, path in enumerate(files)}
        scores_cache = {}
        all_processed_images = []
        history_index = -1
        last_processed_folder = folder_path
        print(f"{len(images_in_folder)}枚の画像が{folder_path}で見つかりました")

    sample = list(images_in_folder)
    random.shuffle(sample)
    return sample[:100]


def process_images(folder_path: str, threshold: float) -> None:
    global current_showing_index, is_running, scores_cache, all_processed_images, history_index, images_in_folder, images_in_folder_index, last_processed_folder

    print(f"処理開始: {folder_path}")
    current_images = update_stack(folder_path)

    for image_path in current_images:
        try:
            print(f"Processing {image_path}")
            image = get_image(image_path)
            score = get_score(image)
            print(f"{image_path}: {score:.2f}")
            scores_cache[image_path] = score
            all_processed_images.append((image_path, score))
            history_index = len(all_processed_images) - 1
            current_showing_index = images_in_folder_index.get(image_path, -1)
            yield image_path, image, score

            if score >= threshold or not is_running:
                is_running = False
                break
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


def gradio_interface(folder_path, threshold):
    global is_running
    is_running = True
    
    if not os.path.isdir(folder_path):
        yield gr.update(value="指定されたフォルダが存在しません。"), gr.update(), gr.update(), gr.update()
        return

    image_generator = process_images(folder_path, threshold)
    
    for image_path, image, score in image_generator:
        if not is_running:
            break
        if score >= threshold:
            yield (
                image,
                image_path,
                f"{score:.2f}",
                f"閾値以上のスコアの画像が見つかりました: {image_path}"
            )
            return
        else:
            yield (
                image,
                image_path,
                f"{score:.2f}",
                f"評価中: {image_path} (スコア: {score:.2f})"
            )

    yield (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update()
    )
def copy_image(image_path: str) -> str:
    """
    指定された画像パスの画像を'good'フォルダにコピーする関数。

    RAWファイルが存在する場合は、RAWファイルをコピーします。
    存在しない場合は、元の画像ファイルをコピーします。

    Args:
        image_path (str): コピーする画像のパス

    Returns:
        str: コピー操作の結果を示すメッセージ

    """
    if image_path:
        # 'good'フォルダを作成（既に存在する場合は何もしない）
        good_folder = "good"
        os.makedirs(good_folder, exist_ok=True)

        # RAWファイルの拡張子リスト
        raw_suffix = [".NEF", ".ARW"]
        original_path = image_path

        # RAWファイルが存在するか確認
        for suffix in raw_suffix:
            raw_path = os.path.splitext(image_path)[0] + suffix
            if os.path.exists(raw_path):
                original_path = raw_path
                break
        
        # コピー先のパスを生成
        dest_path = os.path.join(good_folder, os.path.basename(original_path))


        def get_file_hash(file_path):
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
            return file_hash.hexdigest()

        if os.path.exists(dest_path):
            original_hash = get_file_hash(original_path)
            dest_hash = get_file_hash(dest_path)
            
            if original_hash == dest_hash:
                return f"同一ファイルが既に存在します: {dest_path}"
            
            base, ext = os.path.splitext(dest_path)
            i = 2
            while os.path.exists(f"{base}-{i}{ext}"):
                i += 1
            dest_path = f"{base}-{i}{ext}"
            shutil.copy(original_path, dest_path)
        else:
            shutil.copy(original_path, dest_path)

        # コピー成功メッセージを返す
        return f"画像を'good'フォルダにコピーしました: {dest_path}"
    
    # 画像パスが指定されていない場合のエラーメッセージ
    return "画像がありません。"

def browse_folder():
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
        "$d.Description = 'フォルダを選択';"
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True
    )
    folder = result.stdout.strip()
    return folder if folder else gr.update()

def stop_evaluation():
    global is_running
    is_running = False
    return "評価を停止しました。"

def _show_folder_image(index):
    image_path = images_in_folder[index]
    image = get_image(image_path)
    score = scores_cache.get(image_path)
    if score is None:
        score = get_score(image)
        scores_cache[image_path] = score
    return image, image_path, f"{score:.2f}"

def previous_image():
    global current_showing_index, is_running
    is_running = False
    if not images_in_folder:
        return gr.update(), gr.update(), gr.update(), "画像が読み込まれていません。"
    if current_showing_index > 0:
        current_showing_index -= 1
        image, path, score_str = _show_folder_image(current_showing_index)
        return image, path, score_str, f"前の写真: {path}"
    return gr.update(), gr.update(), gr.update(), "これ以上前の画像はありません。"

def next_image():
    global current_showing_index, is_running
    is_running = False
    if not images_in_folder:
        return gr.update(), gr.update(), gr.update(), "画像が読み込まれていません。"
    if current_showing_index < len(images_in_folder) - 1:
        current_showing_index += 1
        image, path, score_str = _show_folder_image(current_showing_index)
        return image, path, score_str, f"次の写真: {path}"
    return gr.update(), gr.update(), gr.update(), "これ以上次の画像はありません。"

def previous_history():
    global history_index, current_showing_index, is_running
    is_running = False
    if not all_processed_images:
        return gr.update(), gr.update(), gr.update(), "評価済み画像がありません。"
    if history_index > 0:
        history_index -= 1
        path, score = all_processed_images[history_index]
        current_showing_index = images_in_folder_index.get(path, current_showing_index)
        image = get_image(path)
        return image, path, f"{score:.2f}", f"履歴 ({history_index + 1}/{len(all_processed_images)}): {path}"
    return gr.update(), gr.update(), gr.update(), "履歴の先頭です。"

def next_history():
    global history_index, current_showing_index, is_running
    is_running = False
    if not all_processed_images:
        return gr.update(), gr.update(), gr.update(), "評価済み画像がありません。"
    if history_index < len(all_processed_images) - 1:
        history_index += 1
        path, score = all_processed_images[history_index]
        current_showing_index = images_in_folder_index.get(path, current_showing_index)
        image = get_image(path)
        return image, path, f"{score:.2f}", f"履歴 ({history_index + 1}/{len(all_processed_images)}): {path}"
    return gr.update(), gr.update(), gr.update(), "履歴の末尾です。"

# Gradio インターフェース
with gr.Blocks() as demo:
    # gr.Markdown("# 美的画像セレクター")
    with gr.Row():
        with gr.Column(scale=3):
            folder_path = gr.Textbox(label="フォルダパス", value=default_folder_name, placeholder="画像フォルダのパスを入力してください")
            browse_button = gr.Button("📁 フォルダ選択", size="sm", elem_id="browse-btn")
        threshold = gr.Slider(minimum=4, maximum=6, value=5, label="閾値", scale=1)
    
    with gr.Row():
        start_button = gr.Button("評価開始")
        stop_button = gr.Button("評価停止")
        good_button = gr.Button("保存")
    with gr.Row():
        prev_history_button = gr.Button("◀ 履歴")
        next_history_button = gr.Button("履歴 ▶")
        previous_button = gr.Button("◀ 前後")
        next_button = gr.Button("前後 ▶")
    
    current_image_path = gr.Textbox(visible=False)

    with gr.Column():
        image = gr.Image(label="現在の画像", interactive=False, height=1000)
        score = gr.Textbox(label="美的スコア")
        status = gr.Textbox(label="ステータス")
    
    output = gr.Textbox(label="アクション結果")

    browse_button.click(
        browse_folder,
        outputs=[folder_path]
    )

    start_button.click(
        gradio_interface,
        inputs=[folder_path, threshold],
        outputs=[image, current_image_path, score, status]
    )
    
    good_button.click(
        copy_image,
        inputs=[current_image_path],
        outputs=[output]
    )

    stop_button.click(
        stop_evaluation,
        outputs=[status]
    )

    prev_history_button.click(
        previous_history,
        outputs=[image, current_image_path, score, status]
    )

    next_history_button.click(
        next_history,
        outputs=[image, current_image_path, score, status]
    )

    previous_button.click(
        previous_image,
        outputs=[image, current_image_path, score, status]
    )

    next_button.click(
        next_image,
        outputs=[image, current_image_path, score, status]
    )

demo.queue().launch(debug=True, css="#browse-btn { max-width: 130px; }")