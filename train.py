# ... 既存のインポート文 ...
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
import datetime
import random
random.seed(42)

state_name = "sac+logos+ava1-l14-linearMSE.pth"

all_photo_folder = "/mnt/d/photo/camera/"
good_photo_folder = "./photos_in_hp"

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
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
pt_state = torch.load(state_name)
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


def evaluate_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(np.array(X)).to(device).float()
        y_tensor = torch.from_numpy(np.array(y)).to(device).float().unsqueeze(1)
        outputs = model(X_tensor)
        loss = nn.MSELoss()(outputs, y_tensor)
    return loss.item()

def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0)  # GPUに保持したまま、1次元テンソルとして返す

def retrain_model(good_images):
    import time 
    start_time = time.time()
    print("データの準備")
    # データの準備
    X = []
    y = []
    for image_path in good_images:
        img = Image.open(image_path)
        features = get_image_features(img)
        X.append(features.cpu().numpy())  # CPUに移動してからNumPy配列に変換
        y.append(6.0)  # 良い画像は1.0


    # データを訓練セットと検証セットに分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルの再訓練
    predictor.train()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 100
    batch_size = 32
    patience = 10  # early stoppingの閾値
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # all_photos = glob.glob(f"{all_photo_folder}/**/*.{{jpg,JPG}}", recursive=True)
    all_photos = []
    for ext in ["jpg", "JPG", "jpeg", "JPEG"]:
        all_photos.extend(glob.glob(f"{all_photo_folder}/**/*.{ext}", recursive=True))

    print(f"{len(all_photos)}枚の画像があります。")

    predictor.to(device)

    print(f"初期処理にかかった時間: {time.time() - start_time}秒")
    start_time = time.time()

    # 学習前のモデルの評価
    initial_train_loss = evaluate_model(predictor, X_train, y_train, device)
    initial_val_loss = evaluate_model(predictor, X_val, y_val, device)
    print(f"学習前 - 訓練損失: {initial_train_loss:.4f}, 検証損失: {initial_val_loss:.4f}")

    for epoch in tqdm(range(num_epochs), desc="エポック"):
        X_train_random_sample = random.sample(X_train, 128)
        y_train_random_sample = [6.0] * len(X_train_random_sample)

        # ネガティブサンプルの追加（エポックごとに変更）
        start_time = time.time()
        random_images = random.sample(all_photos, len(X_train_random_sample))
        X_negative = []
        y_negative = []
        for image_path in tqdm(random_images, desc="ネガティブサンプルの特徴抽出", leave=False):
            try:
                img = Image.open(image_path)
                features = get_image_features(img)
                X_negative.append(features.cpu().numpy())  # CPUに移動してからNumPy配列に変換
                y_negative.append(4.0)  # ランダムな画像は0.0
            except:
                print(f"cannnot open {image_path}, skip")
        print(f"ネガティブサンプルの特徴抽出にかかった時間: {time.time() - start_time}秒")
        start_time = time.time()

        X_epoch = np.array(X_train_random_sample + X_negative)
        y_epoch = np.array(y_train_random_sample + y_negative)

        # データのシャッフル
        indices = np.arange(len(X_epoch))
        np.random.shuffle(indices)
        X_epoch = X_epoch[indices]
        y_epoch = y_epoch[indices]

        total_loss = 0
        num_batches = 0
        for i in range(0, len(X_epoch), batch_size):
            batch_X = torch.from_numpy(X_epoch[i:i+batch_size]).to(device).float()
            batch_y = torch.from_numpy(y_epoch[i:i+batch_size]).to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = predictor(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 検証データでの評価
        predictor.eval()
        with torch.no_grad():
            val_X = torch.from_numpy(np.array(X_val)).to(device).float()
            val_y = torch.from_numpy(np.array(y_val)).to(device).float().unsqueeze(1)
            val_outputs = predictor(val_X)
            val_loss = criterion(val_outputs, val_y)

        predictor.train()

        avg_train_loss = total_loss / num_batches
        print(f"エポック {epoch+1}/{num_epochs}, 訓練損失: {avg_train_loss:.4f}, 検証損失: {val_loss:.4f}")

       # Early stopping のチェック
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # ベストモデルの保存
            best_model_state = predictor.state_dict().copy()
            # ベストモデルの保存
            current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f"best_model_valid_loss_{best_val_loss:.4f}_epoch{epoch+1}.pth"
            torch.save(predictor.state_dict(), best_model_path)
            print(f"新しいベストモデルを保存しました: {best_model_path}")

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: {patience} エポック連続で改善が見られませんでした。")
                break


        if torch.cuda.is_available():
            print(f"GPU使用メモリ: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")



    predictor.eval()

    # ベストモデルの読み込み
    predictor.load_state_dict(best_model_state)

    # モデルの保存（日付を含む）
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    model_save_path = f"retrained_model_{current_date}.pth"
    torch.save(predictor.state_dict(), model_save_path)
    print(f"モデルを保存しました: {model_save_path}")
# メイン処理
if __name__ == "__main__":

    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    print(f"現在のデバイス: {device}")
    if torch.cuda.is_available():
        print(f"現在のCUDAデバイス: {torch.cuda.current_device()}")
        print(f"デバイス名: {torch.cuda.get_device_name(0)}")

    # 初期重みの読み込み
    predictor.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth"))
    predictor.to(device)

    # 良い画像のパスを取得
    

    good_images = glob.glob(f"{good_photo_folder}/*.jpg")
    print(f"{len(good_images)}枚の良い画像が選択されました。モデルを再訓練します。")

    # モデルの再訓練
    retrain_model(good_images)