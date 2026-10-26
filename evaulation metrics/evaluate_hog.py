import sys,time,argparse
from pathlib import Path
import cv2,joblib,numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import config
from eval_common import CLASSES,save_json,plot_matrix,plot_class_metrics

def feature(path):
    img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if img is None:raise ValueError(f'Cannot read {path}')
    if img.shape!=(64,64):img=cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
    return hog(img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True).astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='test',choices=['val','test']);a=p.parse_args()
    root=config.CLASSIFICATION_PROCESSED/a.split;model_path=config.HOG_RESULTS_DIR/'hog_svm_model.joblib'
    if not model_path.is_file():raise FileNotFoundError(model_path)
    paths=[f for name in CLASSES for f in sorted((root/name).glob('*.png'))]
    if not paths:raise FileNotFoundError(f'No processed PNG images in {root}')
    y=[f.parent.name for f in paths];model=joblib.load(model_path)
    # Bounded chunks prevent huge RAM usage.
    pred=[];duration=0.
    for i in range(0,len(paths),256):
        X=np.stack([feature(f) for f in paths[i:i+256]])
        t=time.perf_counter();pred.extend(model.predict(X).tolist());duration+=time.perf_counter()-t
    report=classification_report(y,pred,labels=CLASSES,output_dict=True,zero_division=0)
    cm=confusion_matrix(y,pred,labels=CLASSES)
    out=config.HOG_RESULTS_DIR/'evaluation'/a.split;out.mkdir(parents=True,exist_ok=True)
    metrics={'model':'HOG + Linear SVM','split':a.split,'images':len(y),'accuracy':float(accuracy_score(y,pred)),
             'macro_f1':float(report['macro avg']['f1-score']),'weighted_f1':float(report['weighted avg']['f1-score']),
             'macro_precision':float(report['macro avg']['precision']),'macro_recall':float(report['macro avg']['recall']),
             'fps_predict_only':len(y)/duration if duration else None,'per_class':{n:{'precision':report[n]['precision'],'recall':report[n]['recall'],'f1':report[n]['f1-score'],'support':int(report[n]['support'])} for n in CLASSES}}
    save_json(out/'metrics.json',metrics);plot_matrix(cm,out/'confusion_matrix.png',CLASSES);plot_class_metrics(metrics['per_class'],out/'per_class_metrics.png')
    print(metrics);print('Saved to',out)
if __name__=='__main__':main()
