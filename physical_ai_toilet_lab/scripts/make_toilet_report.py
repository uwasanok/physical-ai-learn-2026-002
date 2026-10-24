"""Submission report from measured artifacts; local Windows bundled Python."""
import json
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/submission'
runs=json.loads((ROOT/'output/closed_loop_v1/summary.json').read_text())
assert all(r['remaining_cells']==0 and not r['false_clear'] for r in runs[:3])
pdfmetrics.registerFont(TTFont('JP','C:/Windows/Fonts/msgothic.ttc',subfontIndex=0))
styles={k:ParagraphStyle(k,fontName='JP',fontSize=size,leading=lead,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#163148')) for k,size,lead in [('title',20,28),('h',12,19),('p',10,16),('small',8,12)]}
story=[]
def p(s,style='p'):story.append(Paragraph(s,styles[style]))
def table(rows,widths):
    t=Table([[Paragraph(str(x),styles['small']) for x in row] for row in rows],colWidths=widths)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0f1ef')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#b0c6d0')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,10))
def page(canvas,doc):
    canvas.setFont('JP',8);canvas.drawString(44,25,'PAI最終課題_53772 | シミュレーション検証');canvas.drawRightString(550,25,str(doc.page))

p('画像AIで汚れを見つけ、<br/>曲面を清掃して見直すロボット','title')
p('Physical AI 応用1 最終課題 / OmniCampus ID: 53772 / 2026-10-24','small')
p('1. 目的と対象','h')
p('会社のトイレを想定し、大便器の内側の一部分に付いた汚れを、固定筐体上の俯瞰カメラとブラシ付きUR5eアームで清掃する。移動・便座開閉・工具交換を省き、曲面上のランダムな汚れと、複数回こすって残りを確認する動作に絞った。')
p('2. AIとアルゴリズムの役割','h')
p('RGB画像 → 小型U-Netによる汚れ領域の推論 → 校正済みカメラと既知曲面による3D座標変換 → 工具位置・面の向き → 逆運動学 → 衝突検査 → 関節角度の実行 → 撮影姿勢へ退避して再認識。次の清掃位置は毎回の画像から決め、汚れの正解座標や必要回数nは計画へ渡さない。')
table([['学習したもの','数式・アルゴリズムで処理するもの','今回扱わないもの'],['画像から汚れ画素を予測するU-Net','3D位置、面の法線、関節角度、経路、衝突検査','動作方策の模倣・強化学習、トルク制御、実機']],[155,185,165])
story.append(Image(str(OUT/'video_preview_1.png'),width=505,height=284))
p('図1：記録した関節軌道の再生。右側は最後の固定カメラ画像、ピンクはAIの検出。左の残留汚れ表示は評価用で、制御入力には使わない。','small')
story.append(PageBreak())
p('3. 実装・学習条件','h')
p('Pythonによる独自の運動学シミュレーション。公式UR5e URDF・衝突形状メッシュ、SciPyの数値逆運動学、FCLの形状干渉検査を使用した。便器は底・内壁・縁を持つ仮形状。260 mmの工具、固定した可視範囲で評価する。ROS/Gazeboで実行したStep 1とは別実装である。')
p('ローカルRTX 4090、Docker、PyTorch 2.9.1 / CUDA 12.6を使用。320×240の合成RGB画像をtrain 600、validation 120、test 120に分割。色・位置・形・照明を変え、シーンseedを分離した。122,817パラメータのU-Netを30 epoch学習し、validationで26 epoch目としきい値0.4を選択。設定凍結後にtestを1回評価した。')
table([['未使用test 120枚','単純な茶色判定','学習モデル'],['IoU（領域の重なり）','5.711%','99.992%'],['再現率','5.711%','99.994%'],['汚れなし25枚での誤検出','0枚','0枚']],[235,135,135])
p('同じ生成器・仮便器・固定カメラでの結果。合成画像では見分けが容易な可能性があり、実画像や別便器への汎化を証明しない。比較対象は従来の単純な茶色判定である。','small')
p('4. 学習モデルを使った動作検証','h')
table([['条件','清掃回数','終了時の残留','終了状態']]+[[r['name'],r['actions'],r['remaining_cells'],'画像上クリア' if r['status']=='observed_roi_clear' else '上限で停止'] for r in runs],[175,80,100,150])
p('3つの新規配置は、汚れごとの必要回数n=1・3・5を混在させ、全例で残留ゼロ。汚れなしでは動かず、n=999・上限3回の条件では未完了として停止した。画像だけで終了を判定し、評価用真値でも残留を照合した。モデルの重み・しきい値は変更していない。')
p(f'全5条件の計画時に計{sum(r["checked_configurations"] for r in runs):,}姿勢を検査。最大関節刻み0.25度以下。保存した全実行姿勢の再検査を含む40テストが成功した。接近・退避時を含む条件付きブラシ接触、隣接部品、固定部には明示的な例外がある。連続時間の無衝突保証ではない。','small')
story.append(PageBreak())
p('5. 講義内容をどこに活かしたか','h')
table([['講義で学んだ考え方','今回の対応'],['順運動学・逆運動学、タスク空間と関節空間','曲面のxyzと工具姿勢から関節角度を計算し、順運動学で接触位置を評価'],['観測と行動のフィードバック','撮影 → 認識 → 清掃 → 再撮影。必要回数を知らずに残りを確認'],['データの多様性と品質','色・位置・照明を変えた教師データ、シーン単位の分割、未使用画像で評価'],['上位の判断と下位制御の分離','画像AIと経路計算を分離。実機の追従制御・トルク層は未実装']],[185,320])
p('関節角度は「どれだけ曲げるか」、トルクは「どれだけ回す力を出すか」。両者は別であり、今回学習したのは画像認識だけである。講義の運動学と制御の分離を活用したが、模倣学習・強化学習・VLAを実装したとは主張しない。課題資料はAIの推論のみでも可、学習は任意としている。')
p('6. 独自の工夫と限界','h')
p('曲面の法線に工具を合わせる、縁と手首の干渉を工具長で比較する、撮影時はアームを退避して遮蔽を避ける、1回で落ちない汚れを再観測する、落ちない場合は試行上限で止める点を組み合わせた。')
p('接触は距離・向き・移動量・局所通過回数による簡易モデル。剛体動力学、追従誤差、ブラシの変形、押付け力、水・洗剤・除菌効果は未評価。実機化には力・トルク制御、実カメラ校正、実物データ、防水・衛生設計などを別途検証する必要がある。')
p('7. 提出動画・コード・参考','h')
p('動画は95秒。約580.48秒の記録から5区間を抜粋し、区間内2倍速、区間間の省略を明示した。冒頭10秒は説明、最後は最終撮影5秒と結果説明5秒。動画は検証ログの可視化であり、実機撮影やGazeboの映像ではない。')
p('コード・学習済みモデル・合成データ・検証ログはGitHubで公開済み。再現手順はリポジトリ内のREPRODUCE_SUBMISSION.mdに記載。動画はYouTubeへアップロード済み。')
p('公開コード：<link href="https://github.com/uwasanok/physical-ai-learn-2026-002" color="#167a93">https://github.com/uwasanok/physical-ai-learn-2026-002</link><br/>提出動画：<link href="https://youtu.be/j69k0yDYFY0" color="#167a93">https://youtu.be/j69k0yDYFY0</link>','small')
p('参照：配布「最終課題（10/19改訂版）」、基礎編第2回（運動学・制御）、第6回（データ・観測と行動）、第7回（方策と下位制御）の手元文字起こし。応用編全講義の確認を意味しない。','small')
p('公式モデル：github.com/UniversalRobots/Universal_Robots_ROS2_Description / jazzy / commit 7ce98d8c3512b9d0239da93b75de68ea64f61bd6。BSD-3-Clause表示を同梱。PyTorch：pytorch.org/get-started/previous-versions/。コード・説明資料の作成には生成AI（Codex）を使用した。','small')
SimpleDocTemplate(str(OUT/'説明資料_PAI最終課題_53772.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=42).build(story,onFirstPage=page,onLaterPages=page)
