import sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import config
sources={'HOG + SVM':config.HOG_RESULTS_DIR/'evaluation/test/metrics.json','YOLOv8':config.ROOT/'evaluation_results/yolo/test/metrics.json','Faster R-CNN':config.ROOT/'evaluation_results/frcnn/test/metrics.json'}
values={name:json.loads(path.read_text()) for name,path in sources.items() if path.is_file()}
if not values:raise SystemExit('No evaluation results found; run evaluation scripts first.')
fig,ax=plt.subplots(figsize=(9,5))
names=list(values);scores=[values[n]['macro_f1'] for n in names]
ax.bar(names,scores);ax.set(ylim=(0,1.05),ylabel='Macro F1',title='Model evaluation (different task definitions; interpret cautiously)')
for i,v in enumerate(scores):ax.text(i,v+.02,f'{v:.3f}',ha='center')
fig.tight_layout();out=config.ROOT/'evaluation_results';out.mkdir(exist_ok=True);fig.savefig(out/'macro_f1_comparison.png',dpi=160);plt.close(fig)
print('Saved',out/'macro_f1_comparison.png')
for n,v in values.items():print(n, 'macro F1:',round(v['macro_f1'],4),'mAP50:',v.get('map50','N/A'))
