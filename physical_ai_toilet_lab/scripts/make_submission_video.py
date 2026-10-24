"""Render measured kinematic logs, never fabricate trajectories. Five 2x excerpts."""
import base64,io,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/submission';OUT.mkdir(exist_ok=True)
s=(ROOT/'output/closed_loop_v1/learned_cleaning_3d.html').read_text(encoding='utf-8')
data=json.loads(s.split('const data=')[1].split(';\nconst canvas')[0]);frames=data['frames'];times=np.array([f['time_s'] for f in frames])
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={n:ImageFont.truetype(font,n) for n in [18,22,26,34]}
def text(draw,xy,s,n=22,color='#e6f1f4'):draw.text(xy,s,font=fonts[n],fill=color)
def base():
    image=Image.new('RGB',(1280,720),'#10202d');draw=ImageDraw.Draw(image)
    text(draw,(28,18),'画像AIで見て、曲面をこすり、残りを見直す',34)
    text(draw,(30,66),'PAI最終課題_53772  |  UR5e / 固定俯瞰カメラ / シミュレーション',18)
    return image,draw
def project(p):
    q=np.asarray(p)-[0,-.2,.45];yaw=-.65;pitch=.65
    x=np.cos(yaw)*q[...,0]-np.sin(yaw)*q[...,1];y=np.sin(yaw)*q[...,0]+np.cos(yaw)*q[...,1]
    z=np.cos(pitch)*q[...,2]-np.sin(pitch)*y;depth=np.cos(pitch)*y+np.sin(pitch)*q[...,2]
    return np.stack([420+x*430,365-z*430,depth],-1)
meshes=[(m['link'],np.array(m['triangles'])) for m in data['arm_meshes']]
static=np.concatenate([np.array(data['bowl_mesh']),np.array(data['rim_mesh'])])
captures=[Image.open(io.BytesIO(base64.b64decode(c['image'].split(',')[1]))).convert('RGB') for c in data['camera_captures']]
phase_names={'observe':'撮影・AIによる汚れ検出','observation_move':'撮影姿勢との往復','approach':'曲面へ接近','clean':'曲面に沿って清掃','retract':'便器から退避'}
def render(t,clip):
    f=frames[max(0,np.searchsorted(times,t,side='right')-1)]
    image,draw=base();draw.rounded_rectangle((20,110,830,615),12,fill='#eaf0f3')
    groups=[(static,np.array([214,227,231]))]
    for name,tri in meshes:
        tf=np.array(f['arm'][name]);groups.append((tri@tf[:3,:3].T+tf[:3,3],np.array([91,147,172])))
    polygons=[]
    for tri,color in groups:
        p=project(tri);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
        colors=(color*(.6+.4*np.abs(n[:,2]))[:,None]).astype(int)
        for pts,c in zip(p,colors):polygons.append((pts[:,2].mean(),pts[:,:2],tuple(c)))
    for _,pts,c in sorted(polygons,key=lambda a:a[0]):draw.polygon([tuple(x) for x in pts],fill=c)
    tip=np.array(f['xyz']);axis=np.array(f['rotation'][2]);a,b=project(np.array([tip,tip+.26*axis]))[:,:2]
    draw.line([tuple(a),tuple(b)],fill='#198e6e',width=5);draw.ellipse((a[0]-6,a[1]-6,a[0]+6,a[1]+6),fill='#14b889')
    for xyz in np.array(data['dirt'])[f['remaining']]:
        p=project(xyz);draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill='#905326')
    obs=max(0,f['observation']);image.paste(captures[obs].resize((400,300)),(855,115))
    text(draw,(855,427),'最後の俯瞰画像 / ピンク＝AI検出',18)
    text(draw,(855,466),f'清掃 {f["action"]} / {data["metrics"]["cleaning_actions"]} 回',26)
    text(draw,(855,509),f'残留区画：{len(f["remaining"])}（評価値）',22)
    text(draw,(855,552),'必要回数nはAI・計画に渡さない',18)
    text(draw,(30,630),f'抜粋 {clip}/5・区間内2倍速・区間間は省略  |  予定時刻 {t:.1f} 秒',22)
    text(draw,(30,670),phase_names.get(f['phase'],f['phase'])+'  /  記録した関節軌道の再生・力学とトルクは未評価',18)
    return image
def title_card(end=False):
    image,draw=base()
    lines=(['検証結果：新しい3配置、すべて残留ゼロ','清掃回数：23回 / 20回 / 36回','汚れなし：0回。落ちない設定：上限で停止。','画像AIを学習。関節角度は逆運動学で計算。','仮の便器・簡易接触モデル。実機の除去・除菌性能は未検証。'] if end else
           ['対象：便器内側の一部分。移動は扱わない。','画像 → 学習モデル → 汚れ領域 → 3D位置','3D位置・面の向き → 逆運動学 → 検査済み経路','清掃後は退避して再撮影。残りがあれば繰り返す。','動画は5場面の抜粋。区間内2倍速・区間間は省略。'])
    for i,line in enumerate(lines):text(draw,(60,170+i*78),line,26 if i<4 else 22)
    text(draw,(60,640),'Pythonによる運動学シミュレーション / ROS・Gazeboでの実行映像ではありません',18)
    return image

def main():
    fps=5;ends=times[-1];starts=[float(times[0]),100.,250.,400.,float(ends-30)]
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-r','10','-movflags','+faststart',str(OUT/'PAI最終課題_53772.mp4')],stdin=subprocess.PIPE)
    for _ in range(10*fps):process.stdin.write(title_card().tobytes())
    for clip,start in enumerate(starts,1):
        for i in range(15*fps):
            image=render(start+2*i/fps,clip);process.stdin.write(image.tobytes())
            if i==35:image.save(OUT/f'video_preview_{clip}.png')
        print(f'Rendered clip {clip}/5',flush=True)
    final=render(float(ends),5);draw=ImageDraw.Draw(final)
    draw.rectangle((20,620,1270,660),fill='#10202d');text(draw,(30,630),'最終撮影：AI検出ゼロ・評価用真値も残留ゼロ（静止表示）',22)
    final.save(OUT/'video_final.png')
    for _ in range(5*fps):process.stdin.write(final.tobytes())
    for _ in range(5*fps):process.stdin.write(title_card(True).tobytes())
    process.stdin.close();assert process.wait()==0
    (OUT/'video_edit.json').write_text(json.dumps(dict(duration_s=95,source='saved online joint trajectory',speed=2,
        segments=[dict(source_start_s=t,source_end_s=t+30) for t in starts],cuts='gaps omitted between segments',
        intro_s=10,final_still_s=5,outro_s=5,render_fps=5,encoded_fps=10,real_robot=False,dynamics=False),indent=2))

if __name__=='__main__':main()
