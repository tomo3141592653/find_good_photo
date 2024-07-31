# ... 既存のインポート文 ...
import keyboard
from sklearn.model_selection import train_test_split
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
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 確率として出力するためにSigmoid関数を追加
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
    return score.item() * 10  # 0-1の確率を0-10のスコアに変換

def show_and_evaluate_image(image_path, score):
    img = Image.open(image_path)
    img.thumbnail((800, 800))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Score: {score:.2f}\n{image_path}")
    plt.show(block=False)
    
    while True:
        if keyboard.is_pressed('right'):
            plt.close()
            return True
        elif keyboard.is_pressed('left'):
            plt.close()
            return False

def retrain_model(good_images):
    # 良い画像のデータを準備
    X = []
    y = []
    for image_path in good_images:
        img = Image.open(image_path)
        features = get_image_features(img)
        X.append(features[0])
        y.append(1.0)  # 良い画像は1.0

    # ランダムな画像も追加（悪い例として）
    random_images = random.sample(glob.glob(f"{folder_path}/**/*.jpg", recursive=True), len(good_images))
    for image_path in random_images:
        img = Image.open(image_path)
        features = get_image_features(img)
        X.append(features[0])
        y.append(0.0)  # ランダムな画像は0.0

    X = np.array(X)
    y = np.array(y)

    # データを訓練セットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルを再訓練
    predictor.train()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(100):  # エポック数は調整可能
        optimizer.zero_grad()
        outputs = predictor(torch.from_numpy(X_train).to(device).float())
        loss = criterion(outputs, torch.from_numpy(y_train).to(device).float().unsqueeze(1))
        loss.backward()
        optimizer.step()

    predictor.eval()

    # チェックポイントを保存
    torch.save(predictor.state_dict(), "retrained_model.pth")

def find_best_photo(folder_path, sample_num=None, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    good_images = []
    swipe_count = 0

    while True:
        # ランダムに画像を選択
        all_images = glob.glob(f"{folder_path}/**/*.jpg", recursive=True)
        random.shuffle(all_images)
        
        for image_path in all_images:
            img = Image.open(image_path)
            score = get_score(img)
            
            if score > 5:  # スコアが5以上の場合のみ表示
                is_good = show_and_evaluate_image(image_path, score)
                if is_good:
                    good_images.append(image_path)
                    swipe_count += 1
                    
                    if swipe_count == 10:
                        print("10枚の良い画像が選択されました。モデルを再訓練します。")
                        retrain_model(good_images)
                        swipe_count = 0
                        good_images = []
                        print("再訓練が完了しました。新しいモデルで続行します。")

if __name__ == "__main__":
    find_best_photo("/mnt/d/photo/camera/2012", output_dir="output/2012")