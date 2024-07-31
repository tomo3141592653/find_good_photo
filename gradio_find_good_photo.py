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
import glob
from tqdm import tqdm
import random

# Define AestheticPredictor class (same as in the original code)
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

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

def process_images(folder_path, threshold, progress=gr.Progress()):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                images.append(os.path.join(root, file))

    print(f"{len(images)} images found in {folder_path}")
    
    random.shuffle(images)

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                print(f"Processing {image_path}")
                score = get_score(img)
                print(f"{image_path}: {score:.2f}")
                yield image_path, score
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def gradio_interface(folder_path, threshold):
    if not os.path.isdir(folder_path):
        yield gr.update(value="指定されたフォルダが存在しません。"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False)
        return

    image_generator = process_images(folder_path, threshold)
    
    for image_path, score in image_generator:
        if score >= threshold:
            yield (
                image_path,  # 画像パス
                image_path,  # 隠れたTextboxに画像パスを保存
                folder_path,  # 元のフォルダパスを保存
                f"{score:.2f}",
                f"閾値以上のスコアの画像が見つかりました: {image_path}",
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
            break  # 評価を一時停止
        else:
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                f"評価中: {image_path} (スコア: {score:.2f})",
                gr.update(interactive=False),
                gr.update(interactive=False)
            )

def evaluate_and_continue(image_path, original_folder_path, threshold, choice):
    if choice == "Good":
        good_folder = "good"
        os.makedirs(good_folder, exist_ok=True)
        dest_path = os.path.join(good_folder, os.path.basename(image_path))
        shutil.copy(image_path, dest_path)
        result = f"画像を'good'フォルダに移動しました: {dest_path}"
    else:
        result = f"画像をスキップしました: {image_path}"

    image_generator = process_images(original_folder_path, threshold)
    
    for next_image_path, score in image_generator:
        if score >= threshold:
            return (
                next_image_path,
                next_image_path,
                original_folder_path,
                f"{score:.2f}",
                f"閾値以上のスコアの画像が見つかりました: {next_image_path}",
                result,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )

    # 閾値以上の画像が見つからなかった場合
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        "評価が完了しました。閾値以上のスコアの画像が見つかりませんでした。",
        result,
        gr.update(interactive=False),
        gr.update(interactive=False)
    )
def update_button_state(image_path):
    if image_path is not None and isinstance(image_path, str) and len(image_path) > 0:
        return gr.update(interactive=True), gr.update(interactive=True)
    return gr.update(interactive=True), gr.update(interactive=True)

# Gradio インターフェースの修正
with gr.Blocks() as demo:
    gr.Markdown("# 美的画像セレクター")
    with gr.Row():
        folder_path = gr.Textbox(label="フォルダパス", value="/mnt/d/photo/camera/2024", placeholder="画像フォルダのパスを入力してください")
        threshold = gr.Slider(minimum=0, maximum=10, value=5, label="閾値")
    start_button = gr.Button("評価開始")

    current_image_path = gr.Textbox(visible=False)  # 隠れたTextboxを追加
    original_folder_path = gr.Textbox(visible=False)  # 元のフォルダパスを保存するための隠れたTextbox
    

    with gr.Column():
        # アップロード機能を削除し、画像表示のみに変更
        image = gr.Image(label="現在の画像", interactive=False)
        current_image_path = gr.Textbox(visible=False)  # 隠れたTextboxを追加
        score = gr.Textbox(label="美的スコア")
        status = gr.Textbox(label="ステータス")
    
    with gr.Row():
        good_button = gr.Button("良い", interactive=False)
        bad_button = gr.Button("悪い", interactive=False)
    
    output = gr.Textbox(label="アクション結果")
    
    start_button.click(
        gradio_interface,
        inputs=[folder_path, threshold],
        outputs=[image, current_image_path, original_folder_path, score, status, good_button, bad_button]
    )
    
    good_button.click(
        evaluate_and_continue,
        inputs=[current_image_path, original_folder_path, threshold, gr.Textbox(value="Good", visible=False)],
        outputs=[image, current_image_path, original_folder_path, score, status, output, good_button, bad_button]
    )

    bad_button.click(
        evaluate_and_continue,
        inputs=[current_image_path, original_folder_path, threshold, gr.Textbox(value="Bad", visible=False)],
        outputs=[image, current_image_path, original_folder_path, score, status, output, good_button, bad_button]
    )

    image.change(
        update_button_state,
        inputs=[image],
        outputs=[good_button, bad_button]
    )

demo.queue().launch()