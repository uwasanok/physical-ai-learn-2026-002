# 提出準備：画像AIによる曲面清掃

提出動画：https://youtu.be/j69k0yDYFY0

公開コード：https://github.com/uwasanok/physical-ai-learn-2026-002

OmniCampusへの送信完了は別途確認します。

## 提出するもの

- PDF：`output/submission/説明資料_PAI最終課題_53772.pdf`
- 動画：`output/submission/PAI最終課題_53772.mp4`（95秒、音声なし、日本語字幕）
- コード：`output/submission/physical_ai_toilet_53772_code.zip`。ZIPを展開してGitHubに置く。
- 理解用の説明：`docs/12_閉ループ検証と提出.md`

## YouTube用のタイトル

PAI最終課題_53772

## YouTube概要欄・提出欄用の説明

```text
画像AIで汚れを見つけ、便器の曲面を清掃して見直すロボット
OmniCampus ID: 53772

固定俯瞰カメラのRGBから小型U-Netが汚れを検出し、既知の便器曲面で3D位置を求めます。
UR5eの公式モデルを使用し、逆運動学と形状衝突検査でブラシの経路を計算します。
清掃後は撮影姿勢へ退避し、汚れが残っていれば繰り返します。
学習したのは画像認識です。関節角度は数値計算し、トルクや動作方策の学習はしていません。

新規3配置は23回・20回・36回で残留ゼロ。汚れなしは動かず、落ちない条件は試行上限で停止。
汚れの必要回数nや正解座標は、画像AIや行動計画に渡していません。

本動画はPythonの運動学シミュレーションで得た関節軌道の再生です。
全95秒：冒頭説明10秒、5区間の抜粋75秒、最終撮影の静止表示5秒、結果説明5秒。
動作区間内は2倍速、区間間は省略しています。元の予定動作時間は約580.48秒です。
実機映像やROS/Gazeboの実行映像ではありません。
便器は仮形状、接触は幾何と通過回数の簡易モデルです。力学・トルク・実物の清掃/除菌効果は未評価。

ローカルRTX 4090、Docker、PyTorch、SciPy、FCLを使用。
公式UR5eモデル： https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
BSD-3-Clauseライセンス表示をコード一式に同梱しています。
コードと説明資料の作成には生成AI（Codex）を使用しています。
```

## 登録先と残作業

GitHub候補： https://github.com/uwasanok/physical-ai-learn-2026-002
既存Step 1の内容と混ぜず、`toilet_lab/` 等の別フォルダへ公開する構成を想定。今回のZIPはその準備用で、まだpushしていない。

提出画面： https://edu.omnicamp.us/courses/172/reports/130/

1. トイレ清掃版のコード一式をpublic GitHubに公開し、URLを確認する。
2. MP4を自身のYouTubeに限定公開以上でアップロードし、再生とURLを確認する。
3. OmniCampusでPDF・GitHub URL・YouTube URLを登録して提出済み表示を確認する。

10/19版の配布資料では締切は2026年11月2日AM10:00。公開・子ども向け・年齢制限等の設定は配布課題資料に従う。
講義資料や文字起こし本体は公開コードに含めない。氏名・所属・メールアドレスも含めない。
