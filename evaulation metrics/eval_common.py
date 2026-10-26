import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ['car', 'bus', 'van', 'others']

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf8')

def match_matrix(gt_boxes, gt_labels, pred_boxes, pred_labels, scores, conf=.5, iou_threshold=.5):
    # Class-agnostic greedy matching; misclassification counts as an off-diagonal match.
    from torchvision.ops import box_iou
    import torch
    cm = np.zeros((5, 5), dtype=np.int64)
    keep = np.flatnonzero(np.asarray(scores) >= conf)
    order = sorted(keep, key=lambda k: -scores[k])
    ious = box_iou(torch.as_tensor(pred_boxes, dtype=torch.float32).reshape(-1,4),
                   torch.as_tensor(gt_boxes, dtype=torch.float32).reshape(-1,4)).numpy()
    used = set()
    for j in order:
        options = [(float(ious[j,k]),k) for k in range(len(gt_labels)) if k not in used and ious[j,k] >= iou_threshold]
        if options:
            _, k = max(options)
            used.add(k)
            cm[int(gt_labels[k]), int(pred_labels[j])] += 1
        else:
            cm[4, int(pred_labels[j])] += 1
    for k in range(len(gt_labels)):
        if k not in used: cm[int(gt_labels[k]), 4] += 1
    return cm

def summarize_matrix(cm):
    per = {}
    for i,name in enumerate(CLASSES):
        tp=float(cm[i,i]); fp=float(cm[:,i].sum()-tp); fn=float(cm[i,:].sum()-tp)
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
        per[name]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'support':int(cm[i,:].sum())}
    return {'per_class':per,'macro_f1':float(np.mean([v['f1'] for v in per.values()])),
            'macro_precision':float(np.mean([v['precision'] for v in per.values()])),
            'macro_recall':float(np.mean([v['recall'] for v in per.values()])),
            'weighted_f1':float(np.average([v['f1'] for v in per.values()],weights=[v['support'] for v in per.values()])) if sum(v['support'] for v in per.values()) else 0.,
            'object_accuracy':float(np.trace(cm[:4,:4])/cm.sum()) if cm.sum() else 0.}

def plot_matrix(cm, out, labels=None):
    labels=labels or CLASSES+['background']; fig,ax=plt.subplots(figsize=(8,7))
    norm=cm/cm.sum(axis=1,keepdims=True).clip(min=1)
    ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel='Predicted',ylabel='Ground truth',title='Confusion matrix (counts; row-normalised colour)')
    for (r,c),v in np.ndenumerate(cm): ax.text(c,r,str(v),ha='center',va='center',color='white' if norm[r,c]>.55 else 'black',fontsize=9)
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right');fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)

def plot_class_metrics(per,out):
    x=np.arange(4);fig,ax=plt.subplots(figsize=(9,5))
    for j,key in enumerate(['precision','recall','f1']):ax.bar(x+(j-1)*.25,[per[n][key] for n in CLASSES],.25,label=key)
    ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1.05),ylabel='Score',title='Per-class metrics');ax.legend();fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)

def coco_evaluate(annotation_path, detections, output):
    output=Path(output);gt=COCO(str(annotation_path))
    # loadRes fails for empty predictions; construct an empty detection dataset.
    if detections: dt=gt.loadRes(detections)
    else:
        dt=COCO();dt.dataset={'images':gt.dataset['images'],'categories':gt.dataset['categories'],'annotations':[]};dt.createIndex()
    ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    precision=ev.eval['precision'];recall=ev.params.recThrs
    fig,ax=plt.subplots(figsize=(8,5));per={}
    for k,name in enumerate(CLASSES):
        p=precision[0,:,k,0,2]; valid=p>=0
        ap50=float(np.mean(p[valid])) if valid.any() else None
        q=precision[:,:,k,0,2];valid2=q>=0
        ap=float(np.mean(q[valid2])) if valid2.any() else None
        per[name]={'ap50':ap50,'ap50_95':ap}
        ax.plot(recall[valid],p[valid],label=f'{name} AP50={ap50:.3f}' if ap50 is not None else name)
    ax.set(xlabel='Recall',ylabel='Precision',ylim=(0,1.05),title='Precision–recall at IoU 0.50');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'precision_recall.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4)
    ax.bar(x-.18,[per[n]['ap50'] or 0 for n in CLASSES],.36,label='AP50')
    ax.bar(x+.18,[per[n]['ap50_95'] or 0 for n in CLASSES],.36,label='AP50:95')
    ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1.05),ylabel='AP',title='Per-class average precision');ax.legend();fig.tight_layout();fig.savefig(output/'per_class_ap.png',dpi=160);plt.close(fig)
    return {'map50':float(ev.stats[1]),'map50_95':float(ev.stats[0]),'per_class_ap':per}

def sample_images(root, images, annotations, predictions, output, confidence=.5, max_images=6):
    rng=np.random.default_rng(42)
    for index in rng.choice(len(images),min(max_images,len(images)),replace=False):
        info=images[int(index)]; img=np.asarray(Image.open(Path(root)/info['file_name']).convert('RGB'))
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        for ax in axs:ax.imshow(img);ax.axis('off')
        axs[0].set_title('Ground truth');axs[1].set_title('Prediction')
        for a in annotations.get(info['id'],[]):
            x,y,w,h=a['bbox'];label=a['category_id']-1
            axs[0].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='lime',linewidth=2));axs[0].text(x,y,CLASSES[label],color='white',bbox={'facecolor':'green'})
        for p in predictions.get(info['id'],[]):
            if p['score']<confidence:continue
            x,y,w,h=p['bbox'];label=p['category_id']-1
            axs[1].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='yellow',linewidth=2));axs[1].text(x,y,f'{CLASSES[label]} {p["score"]:.2f}',color='black',bbox={'facecolor':'yellow'})
        fig.tight_layout();fig.savefig(Path(output)/f'example_{int(index):03}.png',dpi=120);plt.close(fig)
