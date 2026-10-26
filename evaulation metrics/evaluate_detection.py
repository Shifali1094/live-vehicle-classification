import argparse,sys,time,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import config
from eval_common import CLASSES,save_json,match_matrix,summarize_matrix,plot_matrix,plot_class_metrics,coco_evaluate,sample_images

def frcnn_model():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # No download required at evaluation time.
    model=fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=640,max_size=640)
    n=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(n,5)
    model.roi_heads.score_thresh=.001
    model.roi_heads.detections_per_img=100
    return model

def locate_checkpoint(model_name,explicit):
    if explicit:return Path(explicit)
    if model_name=='yolo':
        found=list(config.RUNS_DIR.glob('*/weights/best.pt'))
        if not found:found=list(config.RUNS_DIR.glob('*/weights/last.pt'))
    else:
        found=list(config.FRCNN_DIR.glob('frcnn_epoch*.pt'))+list(config.FRCNN_DIR.glob('frcnn_latest.pt'))+list(config.ROOT.glob('frcnn_epoch*.pt'))+list(config.ROOT.glob('frcnn_latest.pt'))
    if not found:raise FileNotFoundError('No checkpoint found; supply --weights /path/to/checkpoint.pt')
    return max(found,key=lambda p:p.stat().st_mtime)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['yolo','frcnn']);parser.add_argument('--weights')
    parser.add_argument('--split',default='test',choices=['val','test']);parser.add_argument('--conf',type=float,default=.5)
    parser.add_argument('--iou',type=float,default=.5);parser.add_argument('--max-images',type=int,default=0,help='0 = entire split; nonzero = diagnostic subset, not COCO mAP')
    parser.add_argument('--device',default='auto');a=parser.parse_args()
    root=config.LOCALIZATION_PROCESSED;annotation=root/f'coco_{a.split}.json'
    if not annotation.exists():raise FileNotFoundError(annotation)
    data=json.loads(annotation.read_text());images=data['images'][:a.max_images or None]
    if not images:raise ValueError('No images in requested split')
    anns={i['id']:[] for i in images}
    for ann in data['annotations']:
        if ann['image_id'] in anns:anns[ann['image_id']].append(ann)
    ckpt=locate_checkpoint(a.model,a.weights)
    device=('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else a.device
    if a.model=='yolo':
        from ultralytics import YOLO
        model=YOLO(str(ckpt))
    else:
        model=frcnn_model().to(device)
        state=torch.load(ckpt,map_location=device,weights_only=False)
        if isinstance(state,dict) and 'model' in state:state=state['model']
        if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
        model.load_state_dict(state);model.eval()
    out=config.ROOT/'evaluation_results'/a.model/a.split;out.mkdir(parents=True,exist_ok=True)
    detections=[];prediction_map={};cm=np.zeros((5,5),dtype=np.int64)
    seconds=0.
    for index,info in enumerate(images):
        path=root/info['file_name']
        if a.model=='yolo':
            t=time.perf_counter()
            res=model.predict(source=str(path),imgsz=416,conf=.001,iou=.7,max_det=300,device=device,verbose=False)[0]
            if device=='cuda':torch.cuda.synchronize()
            seconds+=time.perf_counter()-t
            boxes=res.boxes.xyxy.cpu().numpy();scores=res.boxes.conf.cpu().numpy();labels=res.boxes.cls.cpu().numpy().astype(int)
        else:
            from torchvision.transforms.functional import to_tensor
            tensor=to_tensor(Image.open(path).convert('RGB')).to(device)
            if device=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode():res=model([tensor])[0]
            if device=='cuda':torch.cuda.synchronize()
            seconds+=time.perf_counter()-t
            boxes=res['boxes'].cpu().numpy();scores=res['scores'].cpu().numpy();labels=res['labels'].cpu().numpy().astype(int)-1
        current=[]
        for box,score,label in zip(boxes,scores,labels):
            if not 0<=label<4:continue
            x1,y1,x2,y2=map(float,box);row={'image_id':info['id'],'category_id':int(label)+1,'bbox':[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],'score':float(score)}
            detections.append(row);current.append(row)
        prediction_map[info['id']]=current
        ground=anns[info['id']]
        gt_boxes=[[v['bbox'][0],v['bbox'][1],v['bbox'][0]+v['bbox'][2],v['bbox'][1]+v['bbox'][3]] for v in ground]
        gt_labels=[v['category_id']-1 for v in ground]
        cm+=match_matrix(gt_boxes,gt_labels,boxes,labels,scores,a.conf,a.iou)
        if (index+1)%100==0:print(f'Processed {index+1}/{len(images)}',flush=True)
    summary=summarize_matrix(cm)
    summary.update({'model':a.model,'split':a.split,'checkpoint':str(ckpt),'images':len(images),'confidence':a.conf,'matching_iou':a.iou,'fps_inference':len(images)/seconds if seconds else None,'seconds_inference':seconds,'subset':bool(a.max_images)})
    if not a.max_images:summary.update(coco_evaluate(annotation,detections,out))
    else:print('Subset mode: COCO mAP omitted (full-split ground truth would invalidate subset scoring).')
    plot_matrix(cm,out/'confusion_matrix.png');plot_class_metrics(summary['per_class'],out/'per_class_metrics.png')
    sample_images(root,images,anns,prediction_map,out,a.conf)
    save_json(out/'metrics.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_class','per_class_ap')},indent=2));print('Saved to',out)
if __name__=='__main__':main()
