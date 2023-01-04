from keras import models,layers,losses,optimizers,metrics,callbacks
from tensorflow_addons.losses import sigmoid_focal_crossentropy
from .models import *
import json
import os
import numpy as np

def model_hybrid(train_dsp,val_dsp,langs,methodname):
    for lang in langs:
        dataset_path=f'datasets/{train_dsp}/{lang}.json'
        with open(dataset_path,'r') as f:
            X_train,Y_train=json.load(f)
            X_train=np.array(X_train)
            Y_train=np.array(Y_train)
            
        model=MyRidgeClassifier()
        print(f'{lang} model fitting...')
        model.fit(X_train,Y_train)
        print(f'finish fit!')

        cand_path=f'candidates/{val_dsp}/{lang}.json'
        with open(cand_path,'r') as f:
            X_test,L_test=json.load(f)
            X_test=np.array(X_test)

        print(f'{lang} model predicting...')
        Y_test=model.predict(X_test)
        print(f'finish predict!')

        for i in range(len(L_test)):
            L_test[i].append(float(Y_test[i]))
        
        last_i=0
        Result=[]
        for i in range(len(L_test)):
            if L_test[last_i][0] != L_test[i][0]:
                rank=sorted(L_test[last_i:i],key=lambda kkv:-kkv[2])
                last_i=i
                Result+=rank[:100]

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
    model_hybrid('train','dev',langs)



    