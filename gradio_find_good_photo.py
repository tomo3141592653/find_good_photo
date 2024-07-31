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

    count = 0
    for image_path in images:
        if count >= 100:
            break
        try:
            with Image.open(image_path) as img:
                print(f"Processing {image_path}")
                score = get_score(img)
                print(f"{image_path}: {score:.2f}")
                yield image_path, score
                count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def gradio_interface(folder_path, threshold):
    if not os.path.isdir(folder_path):
        yield gr.update(value="指定されたフォルダが存在しません。"), gr.update(), gr.update(), gr.update(), gr.update(interactive=False)
        return

    image_generator = process_images(folder_path, threshold)
    
    for image_path, score in image_generator:
        if score >= threshold:
            yield (
                image_path,
                image_path,
                f"{score:.2f}",
                f"閾値以上のスコアの画像が見つかりました: {image_path}",
                gr.update(interactive=True)
            )
            return
        else:
            yield (
                image_path,
                image_path,
                f"{score:.2f}",
                f"評価中: {image_path} (スコア: {score:.2f})",
                gr.update(interactive=False)
            )

    yield (
        gr.update(),
        gr.update(),
        gr.update(),
        "評価が完了しました。閾値以上のスコアの画像が見つかりませんでした。",
        gr.update(interactive=False)
    )

def copy_image(image_path):
    if image_path:
        good_folder = "good"
        os.makedirs(good_folder, exist_ok=True)
        dest_path = os.path.join(good_folder, os.path.basename(image_path))
        shutil.copy(image_path, dest_path)
        return f"画像を'good'フォルダにコピーしました: {dest_path}"
    return "画像がありません。"

# Gradio インターフェース
with gr.Blocks() as demo:
    gr.Markdown("# 美的画像セレクター")
    with gr.Row():
        folder_path = gr.Textbox(label="フォルダパス", value="/mnt/d/photo/camera/2024", placeholder="画像フォルダのパスを入力してください")
        threshold = gr.Slider(minimum=0, maximum=10, value=5, label="閾値")
    start_button = gr.Button("評価開始")
    good_button = gr.Button("良い", interactive=False)
    
    current_image_path = gr.Textbox(visible=False)

    with gr.Column():
        image = gr.Image(label="現在の画像", interactive=False)
        score = gr.Textbox(label="美的スコア")
        status = gr.Textbox(label="ステータス")
    

    output = gr.Textbox(label="アクション結果")
    
    start_button.click(
        gradio_interface,
        inputs=[folder_path, threshold],
        outputs=[image, current_image_path, score, status, good_button]
    )
    
    good_button.click(
        copy_image,
        inputs=[current_image_path],
        outputs=[output]
    )

demo.queue().launch()