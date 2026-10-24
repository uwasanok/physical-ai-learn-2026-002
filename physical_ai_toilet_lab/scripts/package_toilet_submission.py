"""Stage code/data/evidence for public review without uploading anything."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/submission'
def main():
    target=OUT/'physical_ai_toilet_53772_code.zip';manifest=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(ROOT)
            if any(x in rel.parts for x in ['.git','__pycache__','.ipynb_checkpoints','cache']):continue
            if rel.parts[:2]==('output','submission'):continue
            if path.name=='.env':continue
            payload=path.read_bytes()
            if path.suffix in ['.md','.py','.json','.txt','.ps1']:
                try:
                    s=payload.decode('utf-8');s=s.replace(str(ROOT),'./physical_ai_toilet_lab').replace(ROOT.as_posix(),'./physical_ai_toilet_lab')
                    # Remove local personal directory prefixes from older explanatory prose.
                    s=s.replace(str(ROOT.parent)+'\\','./').replace(ROOT.parent.as_posix()+'/','./')
                    payload=s.encode('utf-8')
                except UnicodeDecodeError:pass
            archive.writestr('physical_ai_toilet_lab/'+rel.as_posix(),payload)
            manifest.append(dict(path=rel.as_posix(),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
        base=(ROOT.parent/'physical_ai_step1/Dockerfile').read_bytes()
        archive.writestr('physical_ai_toilet_lab/Dockerfile.base',base)
        readme='''# 再現手順（提出用）

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
'''
        archive.writestr('physical_ai_toilet_lab/REPRODUCE_SUBMISSION.md',readme)
        archive.writestr('physical_ai_toilet_lab/CODE_MANIFEST.json',json.dumps(manifest,indent=2))
    (OUT/'code_package_manifest.json').write_text(json.dumps(dict(files=len(manifest)+3,zip_bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),excluded=['lecture material','git metadata','cache','submission media']),indent=2))
    print(target, target.stat().st_size,flush=True)
if __name__=='__main__':main()
