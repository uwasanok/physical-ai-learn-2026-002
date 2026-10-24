# 便器内側の曲面清掃：検討・実験用ラボ

状態：画像AIと清掃ループの接続・検証が完了。新規3配置で残留ゼロ、汚れなしで動かないこと、落ちない条件で上限停止することを確認。40テスト成功。提出用動画・PDF・コードを準備済み。動力学・トルク制御・連続時間での無衝突保証は未対応。

Step 1とは別のディレクトリです。Step 1のコード・提出物・Git履歴を変更しません。
公開先：https://github.com/uwasanok/physical-ai-learn-2026-002 。提出動画：https://youtu.be/j69k0yDYFY0 。

## 今回の目的

便器前に固定した筐体の上のカメラから便器内側を見て、ランダムな汚れを検出し、
ブラシ付きアームで曲面に沿ってこすり、残りの汚れを確認する。
トイレ内の移動は対象外。まず便器内側の一部分に限定する。

**重点は「曲面上の異なる場所の汚れ」と「複数回こすって残りを確認すること」。接触の物理再現は簡略化します。**

## 読む順番

1. [検討方針](docs/01_検討方針.md)：何を残し、何を省くか
2. [実装と評価の計画](docs/02_実装と評価.md)：着手順・比較実験・完了条件
3. [講義との対応と参照元](docs/03_講義と参照元.md)
4. `output/geometry_study.png`：曲面・カメラ配置・ランダム汚れの予備図
5. [複数回清掃の仕様と実験](docs/04_複数回清掃モデル.md)：最新の合意・実装・限界
6. [カメラで見直して清掃する仕組み](docs/05_カメラから再清掃.md)：今回の成果と再実行方法
7. [3D座標と既製アーム](docs/06_3D座標と既製アーム.md)：xyzによる判定・立体表示・実在アームの候補
8. [UR5eの関節動作](docs/07_UR5eの関節動作.md)：公式モデルの導入、検証結果、未完了の範囲
9. [退避経路と衝突検査](docs/08_退避経路と衝突検査.md)：最新の連続動作・FCL検査・実行方法
10. [便器形状と試験条件](docs/09_便器形状と試験条件.md)：縁の追加、死角、工具長の比較、5配置での検証
11. [撮影姿勢と学習データ](docs/10_撮影姿勢と学習データ.md)：遮蔽を避ける退避、画像・正解マスク、学習前の比較基準
12. [画像モデルの学習](docs/11_画像モデルの学習.md)：RTX 4090での学習、未使用画像での比較、AIとアルゴリズムの役割
13. [閉ループ検証と提出](docs/12_閉ループ検証と提出.md)：新しい3配置でのオンライン動作、講義との対応、再実行・提出資料

## 学習AIで清掃して見直す（最新）

`output/closed_loop_v1/learned_cleaning_3d.html` が最新の3D表示です。
画像AIの認識結果から毎回目標と経路を計算し、順運動学の先端位置で簡易接触を評価して、再撮影します。
新しい3配置は23・20・36回の清掃で残留ゼロ。汚れなしは0回、落ちない設定は3回の上限で停止しました。
`docs/12_閉ループ検証と提出.md` に詳しい結果と限界、`output/submission/` に提出資料があります。

## 画像モデルの学習・評価（接続前の段階）

600枚で小型U-Netを学習し、120枚で設定を選び、未使用の120枚で最終評価しました。
test IoUは従来の茶色判定5.711%、学習モデル99.992%。汚れなし25枚での誤検出は0枚。
固定した合成環境での結果であり、実物や異なる便器の認識性能を示すものではありません。
`output/learning_v1/learning_results.png` に比較図、同フォルダに重み・凍結設定・評価結果を保存しています。
学習したのはRGB画像からの汚れ領域検出。この段階では単体評価、その後の接続結果は上記の閉ループ検証を参照。

## 撮影用の退避と画像データ

`output/observation/observation_cleaning_3d.html` が現在の3D表示です。右側に最後の撮影画像を表示します。
清掃中のアーム遮蔽を確認し、先端を高さ650mmへ上げて横へ退避する撮影姿勢を追加しました。
5配置・318本の撮影姿勢への経路を検査済み。対象ROIの遮蔽は撮影姿勢では0です。

学習用600、調整用120、最終評価用120の計840組を `data/segmentation_v1/` に生成しました。
詳細は同ディレクトリの `DATASET_CARD.md`。学習・評価結果は上の「画像モデルの学習」を参照してください。
画像の例は `output/observation/training_examples.png`、遮蔽比較は `output/observation/occlusion_comparison.png`。

## 内壁・底・縁を追加した形状（撮影退避追加前）

`output/geometry_trials/tool_260mm/toilet_with_rim_3d.html` が現在の3D表示です。
縁を追加すると180mm工具の案では手首が干渉し、同じ範囲・5配置を260mm工具で比較しました。
260mm案では5配置とも経路が得られ、残留ゼロ。全67,072点を再検査し、計28テストが通りました。
メーカーCADではなく仮寸法の設計モデルです。アームによる撮影時の遮蔽はまだ未評価です。
形状・試験結果の図は `output/geometry_trials/geometry_and_trials.png`、詳しい条件は上の09を参照してください。

## 退避・移動を含む旧形状の3D表示

`output/continuous/continuous_cleaning_3d.html` を開いてください。
清掃32回の間を退避・移動・接近の31区間でつなぎ、13,793点で自己衝突・環境衝突を検査しました。
同じ仮形状・seed42での結果です。23テスト成功。詳細な限界は上の08の解説を参照してください。
現在の全テスト実行にはラボ専用Dockerイメージ `physical-ai-toilet:collision` を使います。

## アーム付き3D表示（旧：清掃区間のみ）

`output/ur5e/ur5e_cleaning_3d.html` を開くとUR5eの関節動作を再生できます。
公式URDF・衝突用STLを使用し、seed42の32動作・3,360点を関節角度へ変換しました。
清掃区間間はまだ退避経路がなく、姿勢が切り替わります。実機投入可能な軌道ではありません。
`output/ur5e/ur5e_preview.png` に静止画、`docs/07_UR5eの関節動作.md` に解説があります。

## 3Dで確認する

`output/view3d/toilet_cleaning_3d.html` をブラウザで開くと、曲面・汚れ・ブラシ姿勢・軌道を回転して確認できます。
先端の実際のxyzと軸方向で清掃を判定するよう改善しました。高さがずれた動作では清掃されません。
これは先端だけの旧表示です。アーム付き表示は上の `output/ur5e/ur5e_cleaning_3d.html` を使用してください。

再生成：既存Dockerイメージで `scripts/build_3d_view.py` を実行します。

## カメラから再清掃（最新）

固定俯瞰カメラを数式で再現し、茶色い汚れを画像から検出します。
既知の便器曲面とカメラの校正値で3D位置を求め、短くこすって再撮影します。
必要回数nを制御に渡さず、画像に残っている汚れを繰り返し清掃します。

- `output/camera_loop/observe_clean_repeat.gif`：合成カメラの観測ごとのアニメーション
- `output/camera_loop/camera_progress.png`：清掃前・途中・終了時
- `output/camera_loop/summary.json`：3配置での測定結果
- `scripts/camera_loop.py`：描画・検出・座標変換・再清掃

```powershell
cd ./physical_ai_toilet_lab
docker run --rm --network none --mount "type=bind,source=$PWD,target=/work" -w /work/scripts physical-ai-step1:jazzy python3 -m unittest -v
docker run --rm --network none --mount "type=bind,source=$PWD,target=/work" -w /work/scripts physical-ai-step1:jazzy python3 run_camera_study.py
```

既知形状を使うRGB方式の基準実装です。GazeboのカメラではなくPythonで生成した画像です。
便器は仮の凹面で、縁やアームによる遮蔽、未知形状への対応は未評価です。

## 複数回で落ちる汚れの試作

`scripts/cleaning_model.py` は曲面上の区画ごとに除去に必要な回数nを持たせます。
`scripts/run_cleaning_study.py` は3種類のランダム配置とn=1・3・5の汚れを比較し、
`output/cleaning_study.png` と `output/cleaning_results.json` に出力します。
曲面座標を展開した図であり、カメラ撮影画像ではありません。AIによる認識・アーム動作は次段階です。

```powershell
cd ./physical_ai_toilet_lab
docker run --rm --network none --mount "type=bind,source=$PWD,target=/work" -w /work/scripts physical-ai-step1:jazzy python3 -m unittest -v test_cleaning_model
docker run --rm --network none --mount "type=bind,source=$PWD,target=/work" -w /work/scripts physical-ai-step1:jazzy python3 run_cleaning_study.py
```

## 現時点で動かせるもの

`scripts/geometry_study.py` は寸法を仮置きした楕円放物面と、その面の法線を計算します。
ランダム汚れは表面上に生成します。これは幾何の予備検討で、Gazeboの物理シミュレーションではありません。
図のカメラ配置は仮案で、死角・可到達性・衝突・清掃性能は検証していません。

手元の既存Dockerイメージを使う場合（新しい一時コンテナで実行し、Step 1のコンテナは操作しません）：

```powershell
cd ./physical_ai_toilet_lab
docker run --rm --network none --mount "type=bind,source=$PWD,target=/work" -w /work physical-ai-step1:jazzy python3 scripts/geometry_study.py
```

通常のPython環境では `numpy` と `matplotlib` が必要です。
便器の実寸・CAD・カメラ仕様が決まったら、この仮形状と配置を差し替えます。

## 環境の方針

本実装はROS 2 Jazzy＋Gazebo Harmonicを第一候補にします。
新規コンテナ名・別ポート・別ROS_DOMAIN_IDを使う設計とし、Step 1とデータや制御を混在させません。
GPU学習はこのPCのRTX 4090を優先。Colabは学習単体の代替とし、ROSの接触制御ループとは分離します。
