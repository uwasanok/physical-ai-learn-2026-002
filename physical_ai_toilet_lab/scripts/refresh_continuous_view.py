"""Refresh UI without rerunning an unchanged trajectory computation."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
path=Path(sys.argv[1]) if len(sys.argv)>1 else root/'output/continuous/continuous_cleaning_3d.html'
old=path.read_text(encoding='utf-8')
start=old.index('const data=')+len('const data=');end=old.index(';\nconst canvas',start)
html=(root/'scripts/viewer3d.html').read_text(encoding='utf-8').replace('__SCENE_DATA__',old[start:end])
notice=(root/'output/ur5e/THIRD_PARTY_LICENSE.txt').read_text(encoding='utf-8')
path.write_text(html.replace('</html>','<!--\n'+notice+'\n--></html>'),encoding='utf-8')
