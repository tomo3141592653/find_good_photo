# 美的画像セレクター

このプログラムは、指定されたフォルダ内の画像を自動的に評価し、美的に優れた画像を選別するツールです。CLIPモデルと美的評価モデルを使用して、画像の視覚的な魅力度を数値化し、高品質な画像を効率的に見つけることができます。

## 主な機能

- 指定フォルダ内の画像を自動スキャン
- 画像の美的評価スコアの計算（1-10の範囲）
- 設定した閾値以上のスコアを持つ画像の検出
- 選別された画像の「good」フォルダへの自動コピー
- 直感的なGradioインターフェースによる操作
- RAW形式（ARW、NEF）やHEIC形式の画像にも対応

## 対応画像形式

- JPG, JPEG
- RAW形式 (ARW, NEF)
- HEIC

## インストール方法

### uvを使用する場合（推奨）

1. uvをインストールします（まだの場合）：
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 依存関係をインストールして起動します：
```bash
uv run python gradio_find_good_photo.py
```
初回実行時に自動的に仮想環境が作成され、依存関係がインストールされます。

### pipを使用する場合

1. 必要なライブラリをインストールします：

```bash
pip install numpy pillow requests torch torchvision clip gradio pillow-heif rawpy
```

2. CLIPモデルをインストールします：

```bash
pip install git+https://github.com/openai/CLIP.git
```

3. 美的評価モデルをダウンロードします：

プログラムを初めて実行すると、`sac+logos+ava1-l14-linearMSE.pth`というモデルファイルが自動的にダウンロードされます。

## 使用方法

1. `default_folder_name.txt`に探索するデフォルトのフォルダパスを設定します。

2. プログラムを実行します：

```bash
# uvを使用する場合（推奨）
uv run python gradio_find_good_photo.py

# または、pipを使用する場合
python gradio_find_good_photo.py
```

3. ブラウザで表示されるローカルURLにアクセスします（通常は`http://127.0.0.1:7860`）。

4. インターフェースの操作方法：
   - フォルダパスを入力：評価したい画像が含まれるフォルダのパスを入力
   - 閾値を設定：スライダーを使用して「良い」と判断する画像のスコア閾値を設定（4〜6の範囲）
   - 「評価開始」ボタン：画像の評価プロセスを開始
   - 「評価停止」ボタン：評価プロセスを中断
   - 「前の画像」/「次の画像」ボタン：評価済み画像を閲覧
   - 「保存」ボタン：現在表示されている画像を「good」フォルダにコピー

## exeファイルの作成方法

1. exeファイルを作成します：

```bash
# uvを使用する場合（推奨）
uv run pyinstaller --onefile --noconsole --add-data "default_folder_name.txt;." gradio_find_good_photo.py

# または、pipを使用する場合
pyinstaller --onefile --noconsole --add-data "default_folder_name.txt;." gradio_find_good_photo.py
```

2. 生成されたexeファイルは`dist`フォルダにあります。

注意事項：
- exeファイルの初回実行時に、美的評価モデル（`sac+logos+ava1-l14-linearMSE.pth`）が自動的にダウンロードされます
- exeファイルと同じフォルダに`default_folder_name.txt`を配置してください
- 初回実行時は少し時間がかかる場合があります

## 注意事項

- RAWファイルが存在する場合、JPEGよりも優先してコピーされます
- 同名のファイルが「good」フォルダに既に存在する場合、自動的にリネームされます
- CUDA対応GPUがある場合は自動的にGPUを使用します
- 画像は最大1000x1000ピクセルにリサイズされます

## システム要件

- Python 3.10以上
- CUDA対応GPUを推奨（高速処理のため）
- 十分なストレージ空間（処理する画像の数と大きさに依存）

## 技術的詳細

- CLIPモデル（ViT-L/14）を使用して画像の特徴を抽出
- カスタム訓練された美的評価モデル（多層パーセプトロン）を使用
- Gradioを使用したインタラクティブなウェブインターフェース
- マルチプラットフォーム対応（Windows, Unix系OS）

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。



