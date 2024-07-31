
from pathlib import Path
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import math
import os
from tqdm import tqdm
import random
import rawpy

state_name = "sac+logos+ava1-l14-linearMSE.pth"

if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
pt_state = torch.load(state_name)
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


def show_images(image_scores,output_dir="output"):
    num_images = len(image_scores)
    for index in range(num_images):
        image_path, score = image_scores[index]
        img = Image.open(image_path)
        img.thumbnail((800, 800))
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Score: {score:.2f}\n{image_path}")
        
        output_path = f"{output_dir}/output_image_{index}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Image saved: {output_path}")

def find_best_photo(folder_path,sample_num=None,output_dir="output"):
    os.makedirs(output_dir,exist_ok=True)

    # 画像ファイルを取得
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))

    if sample_num is not None:
        random.shuffle(images)
        images = images[:sample_num]

    # 画像とそのスコアのリストを作成
    image_scores = []
    


    for image in tqdm(images, desc="画像を評価中", unit="枚"):
        if image.lower().endswith('.nef'):
            with rawpy.imread(image) as raw:
                # RAW画像の処理を高速化
                rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8)
                # 8ビットで直接出力するため、変換は不要
               
                pil_image = Image.fromarray(rgb)
        else:
            pil_image = Image.open(image)

        try:
            score = get_score(pil_image)
        except:
            score = 0
        image_scores.append((image, score))

    # スコアの高い順にソート
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # 上位n枚の画像を表示
    show_n = 50

    num_images = min(show_n, len(image_scores))

    # インタラクティブな表示を呼び出し
    show_images(image_scores[:num_images],output_dir=output_dir)
   
if __name__ == "__main__":
    find_best_photo("/mnt/d/photo/camera/2013",output_dir="output/2013")