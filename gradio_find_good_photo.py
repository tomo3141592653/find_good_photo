import gradio as gr
import torch
import clip
import numpy as np
from PIL import Image
import os
import shutil
from pathlib import Path
import requests
import torch.nn as nn
import random
import rawpy

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

device = "cuda" if torch.cuda.is_available() else "cpu"
pt_state = torch.load(state_name, map_location=device)
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


# Global variables to keep track of the current state
current_showing_index = -1 # 現在表示している画像のインデックス
is_running = False # 評価中かどうか
all_processed_images = [] # すべての処理済みの画像
images_in_folder = []
last_processed_folder = ""

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

def rotate_by_exif(image):
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


def get_image(image_path:str):
    """
   
    指定された画像パスから画像を読み込む関数。

    Args:
        image_path (str): 画像ファイルへのパス。

    Returns:
        PIL.Image.Image: 読み込まれた画像オブジェクト。

    Note:
        - JPG、JPEG、PNG形式の画像ファイルを直接開きます。
        - ARWまたはNEF形式（RAWファイル）の場合：
          - 同名のJPGファイルが存在すれば、それを優先して開きます。
          - JPGが見つからない場合、RAWファイルを直接処理します。
    """
    if image_path.lower().endswith(('.arw', '.nef')):
        # JPG版を探す
        jpg_path = os.path.splitext(image_path)[0] + '.JPG'
        if os.path.exists(jpg_path):
            image = Image.open(jpg_path)
            # rotate by exif
            image = rotate_by_exif(image)
            image.thumbnail((1000,1000),Image.LANCZOS)
        else:
            # RAWファイルを直接開いて処理
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True,half_size=True)
            image = Image.fromarray(rgb)
            image.thumbnail((1000,1000),Image.LANCZOS)
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
    current_images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.arw','.nef')):
                current_images.append(os.path.join(root, file))
    return current_images

def process_images(folder_path, threshold):
    global current_showing_index, is_running, all_processed_images, images_in_folder, last_processed_folder
    
    if folder_path != last_processed_folder or not images_in_folder:
        images_in_folder = find_files_in_folder(folder_path)
        last_processed_folder = folder_path
        print(f"{len(images_in_folder)}枚の画像が{folder_path}で見つかりました")
    
    random.shuffle(images_in_folder)
    # 100個サンプル
    current_images = images_in_folder[:100]

    current_showing_index = 0

    for image_path in current_images:
        try:
       
            print(f"Processing {image_path}")
            image = get_image(image_path)
            score = get_score(image)
            print(f"{image_path}: {score:.2f}")
            all_processed_images.append((image_path, score))
            current_showing_index = len(all_processed_images) - 1 # show last image
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
        "評価が完了しました。閾値以上のスコアの画像が見つかりませんでした。"
    )

def copy_image(image_path):
    if image_path:
        good_folder = "good"
        os.makedirs(good_folder, exist_ok=True)

        # もし同名のrawファイルがあればそれをコピーする｡なければそのまま
        raw_suffix = [".NEF",".ARW"]
        original_path = image_path
        for suffix in raw_suffix:
            raw_path = os.path.splitext(image_path)[0] + suffix
            if os.path.exists(raw_path):
                original_path = raw_path
                break
        
        dest_path = os.path.join(good_folder, os.path.basename(original_path))
        shutil.copy(original_path, dest_path)
        return f"画像を'good'フォルダにコピーしました: {dest_path}"
    return "画像がありません。"

def stop_evaluation():
    global is_running
    is_running = False
    return "評価を停止しました。"

def previous_image():
    global current_showing_index,is_running
    is_running = False
    if current_showing_index > 0:
        current_showing_index -= 1
        image_path, score = all_processed_images[current_showing_index]
        image = get_image(image_path)
        return image, image_path, f"{score:.2f}", f"前の画像: {image_path}"
    return gr.update(), gr.update(), gr.update(), "これ以上前の画像はありません。"

def next_image():
    global current_showing_index,is_running
    is_running = False
    if current_showing_index < len(all_processed_images) - 1:
        current_showing_index += 1
        image_path, score = all_processed_images[current_showing_index]
        image = get_image(image_path)
        return image, image_path, f"{score:.2f}", f"次の画像: {image_path}"
    return gr.update(), gr.update(), gr.update(), "これ以上次の画像はありません。"

# Gradio インターフェース
with gr.Blocks() as demo:
    # gr.Markdown("# 美的画像セレクター")
    with gr.Row():
        folder_path = gr.Textbox(label="フォルダパス", value="/mnt/d/photo/camera/2024", placeholder="画像フォルダのパスを入力してください")
        threshold = gr.Slider(minimum=4, maximum=6, value=5, label="閾値")
    
    with gr.Row():
        start_button = gr.Button("評価開始")
        stop_button = gr.Button("評価停止")
        previous_button = gr.Button("前の画像")
        next_button = gr.Button("次の画像")
        good_button = gr.Button("保存")
    
    current_image_path = gr.Textbox(visible=False)

    with gr.Column():
        image = gr.Image(label="現在の画像", interactive=False, height=1000)
        score = gr.Textbox(label="美的スコア")
        status = gr.Textbox(label="ステータス")
    
    output = gr.Textbox(label="アクション結果")

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

    previous_button.click(
        previous_image,
        outputs=[image, current_image_path, score, status]
    )

    next_button.click(
        next_image,
        outputs=[image, current_image_path, score, status]
    )

demo.queue().launch()