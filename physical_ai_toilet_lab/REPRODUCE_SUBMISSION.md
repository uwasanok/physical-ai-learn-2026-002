# 再現手順（提出用）

このZIPは学習済み重み・合成データ・測定ログ・公式UR5eモデル・コードを含みます。
講義のPDFや文字起こし本体は含みません。実機制御コードではありません。

WindowsのDocker DesktopでLinuxコンテナとNVIDIA GPUを利用する構成です。
展開したphysical_ai_toilet_labフォルダで次を実行します（初回buildのみネット接続）。

```powershell
docker build -f Dockerfile.base -t physical-ai-step1:jazzy .
docker build -f Dockerfile -t physical-ai-toilet:collision .
docker build -f Dockerfile.training -t physical-ai-toilet:training .
docker run --rm --gpus all --network none --mount "type=bind,source=$PWD,target=/work" physical-ai-toilet:training python3 learned_closed_loop.py
docker run --rm --gpus all --network none --mount "type=bind,source=$PWD,target=/work" physical-ai-toilet:training python3 -m unittest discover -v
```

Dockerfile.baseは今回検証した環境を構築するための基底定義です。以前のStep 1作業ディレクトリは不要です。
同梱重みを使用するため再学習不要。既存の凍結学習・最終画像評価の上書きは禁止する実装です。
閉ループ再実行はoutput/closed_loop_v1を更新するため、測定結果を保存するには別の展開コピーを使います。
README.mdとdocs/12_閉ループ検証と提出.mdから結果を読めます。
HTMLはブラウザで開ける自己完結形式です。最新はoutput/closed_loop_v1/learned_cleaning_3d.html。
動画生成はpython3 make_submission_video.py（Docker内）。PDF生成はWindowsのPython、reportlab、MS Gothicでmake_toilet_report.py。
GPU学習・推論はこのPCのRTX 4090で検証。新しいPCでのDocker初回構築は未検証です。
UR5eモデルのBSD-3-ClauseライセンスとTHIRD_PARTY_LICENSE.txtを保持してください。
