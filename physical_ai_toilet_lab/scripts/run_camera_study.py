import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from cleaning_model import scene
from camera_loop import Camera, Environment, run_loop

out = Path(__file__).resolve().parents[1]/'output'/'camera_loop'
out.mkdir(parents=True, exist_ok=True)
camera = Camera()
results = []
for seed in [42, 123, 999]:
    model, _ = scene(seed)
    result, frames, trajectories = run_loop(Environment(model, camera), keep_frames=seed == 42)
    result['seed'] = seed
    results.append(result)
    (out/f'run_{seed}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out/f'trajectories_{seed}.npz', xyz_normal=np.asarray(trajectories))
    if frames:
        slides = []
        for index, rgb in enumerate(frames):
            im = Image.fromarray(rgb)
            canvas = Image.new('RGB', (640, 530), 'white'); canvas.paste(im, (0, 50))
            draw = ImageDraw.Draw(canvas)
            rec = result['observations'][index]
            draw.text((12, 8), f'SYNTHETIC RGB | observe -> wipe -> observe | action {index}', fill='black')
            draw.text((12, 27), f'Brown pixels: {rec["detected_pixels"]} | no learned AI / no robot physics', fill='black')
            if rec['target_xyz_m']:
                px, py = camera.project(np.array(rec['target_xyz_m']))
                draw.ellipse((px-7, py+43, px+7, py+57), outline='#39ff80', width=2)
            slides.append(canvas)
        slides[0].save(out/'before.png'); slides[-1].save(out/'after.png')
        slides[0].save(out/'observe_clean_repeat.gif', save_all=True, append_images=slides[1:], duration=450, loop=0)
        # A compact contact sheet showing actual re-observations.
        chosen = np.linspace(0, len(slides)-1, 4).astype(int)
        sheet = Image.new('RGB', (1280, 1060), 'white')
        for i, frame_id in enumerate(chosen): sheet.paste(slides[frame_id], ((i%2)*640, (i//2)*530))
        sheet.save(out/'camera_progress.png')
summary = dict(kind='synthetic_camera_known_CAD_color_threshold', camera_position_m=camera.origin.tolist(),
               resolution=[640, 480], focal_pixels=camera.focal,
               limitations=['no real camera or learned AI', 'illustrative paraboloid, no rim/arm occlusion',
                            'ideal curve-following tool; no IK/force physics', 'development seeds only'],
               runs=[{k:v for k,v in r.items() if k != 'observations'} for r in results])
(out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
