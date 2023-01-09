from .models import *
import json
import os
import numpy as np

def fit_predict(Model,datasetname,train_dsp,val_dsp,langs,methodname,hits):
    for lang in langs:
        X_train=np.loadtxt(f'dataset/{datasetname}/feature/{lang}_{train_dsp}.txt',dtype=np.float64,delimiter=',')
        Y_train=np.loadtxt(f'dataset/{datasetname}/rel/{lang}_{train_dsp}.txt',dtype=np.float64,delimiter=',')
            
        model=Model()
        print(f'{lang} model fitting...')
        model.fit(X_train,Y_train)
        print(f'finish fit!')

        X_test:np.ndarray=np.loadtxt(f'dataset/{datasetname}/candidate/{lang}_{val_dsp}.txt',dtype=np.float64,delimiter=',')

        print(f'{lang} model predicting...')
        Y_test:np.ndarray=model.predict(X_test)
        print(f'finish predict!')

        Y_test=Y_test.tolist()
        with open(f'dataset/{datasetname}/pair/{lang}_{val_dsp}.txt','r') as f:
            L_test=[li[:-1].split(',')for li in f.readlines()]
        for i in range(len(L_test)):
            L_test[i].append(float(Y_test[i]))
        
        last_i=0
        Result=[]
        for i in range(len(L_test)):
            if L_test[last_i][0] != L_test[i][0]:
                rank=sorted(L_test[last_i:i],key=lambda kkv:-kkv[2])
                last_i=i
                Result+=rank[:hits]

        rank=sorted(L_test[last_i:],key=lambda kkv:-kkv[2])
        Result+=rank[:100]
        
        try:
            os.makedirs(f'runfiles/{methodname}/')
        except:
            pass
        with open(f'runfiles/{methodname}/{lang}_{val_dsp}.txt','w') as f:
            rk=0
            for tid,docid,rel in Result:
                rk+=1
                f.write(f'{tid} Q0 {docid} {rk} {rel} {methodname}\n')
if __name__=='__main__':
    langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
    fit_predict(MyRidgeClassifier_NonNeg,'all','train','dev',langs,'default_model_hybrid',100)



    