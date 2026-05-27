[English](README.md) | [Português](README.pt.md) | [日本語](README.ja.md) | [Русский](README.ru.md)

# ✋ Palmeter

**Palmeter** は、Pythonで開発されたリアルタイムのハンドトラッキングツールで、複数の手を検出し、**手のひら同士の距離**を計算します。

![PalmeterDemo](https://github.com/KrishBharadwaj5678/Palmeter/raw/main/PalmeterDemo.png)

## 🌟 機能

| 機能              | 説明                                  |
| --------------- | ----------------------------------- |
| ✋ 複数の手の検出       | MediaPipe を使用してリアルタイムで **2つの手** を検出 |
| 📏 手のひら距離計算     | **手のひら間の距離** を計算                    |
| 🎯 リアルタイム表示     | 距離を画面上にリアルタイム表示                     |
| 🖼️ ビジュアルオーバーレイ | 手のランドマークを描画して分かりやすく表示               |
| 🔁 継続的トラッキング    | フレームごとに手の位置を継続更新                    |
| ⚡ 軽量パフォーマンス     | CPUのみでも低スペックPCで高速動作                 |
| 🧪 拡張しやすい設計     | ジェスチャー認識に拡張しやすいクリーンなコード             |

---

## 🛠️ 技術スタック

| 技術            | 説明                    |
| ------------- | --------------------- |
| 🧠 メディアパイプ  | 手のランドマーク検出            |
| 🎥 OpenCV     | リアルタイムコンピュータビジョンライブラリ |
| ⚙️ CVゾーン    | ハンドトラッキングを簡単化         |
| 🐍 パイソン 3.x | プログラミング言語             |

---

## 🚀 インストール

### 1️⃣ リポジトリをクローン

```bash id="8f0gbi"
git clone https://github.com/KrishBharadwaj5678/Palmeter.git
```

### 2️⃣ プロジェクトディレクトリへ移動

```bash id="k8qj7n"
cd Palmeter
```

### 3️⃣ 依存関係をインストール

```bash id="9y0jnn"
pip install -r requirements.txt
```

### 4️⃣ スクリプトを実行

```bash id="6bhhpk"
python app.py
```
