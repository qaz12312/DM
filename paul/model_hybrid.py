import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.metrics as metrics
import json
import os
import numpy as np

train_dsp=f'dev'
val_dsp=f'dev'

def create_model(N):
    xin=layers.Input(shape=(N,))
    x=layers.Dense(1,activation='sigmoid')(xin)
    model=models.Model(inputs=xin,outputs=x)
    model.compile(
        metrics=[metrics.binary_accuracy],
        loss=losses.binary_crossentropy,
        optimizer=optimizers.Adam()
    )
    return model


langs=['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
for lang in langs:
    dataset_path=f'datasets/{train_dsp}/{lang}.json'
    cand_path=f'candidates/{val_dsp}/{lang}.json'
    with open(dataset_path,'r') as f:
        X_train,Y_train=json.load(f)
        X_train=np.array(X_train)
        Y_train=np.array(Y_train)
    model=create_model(len(X_train[0]))
    model.fit(X_train,Y_train,validation_split=0.3,epochs=10,batch_size=10)
    
    with open(cand_path,'r') as f:
        X_test,L_test=json.load(f)
        X_test=np.array(X_test)
    
    
    Y_test=model.predict(X_test)
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
        os.makedirs(f'data/model_hybrid/')
    except:
        pass
    with open(f'data/model_hybrid/{lang}_{val_dsp}.txt','w') as f:
        rk=0
        for tid,docid,rel in Result:
            rk+=1
            f.write(f'{tid} Q0 {docid} {rk} {rel} model_hybrid\n')




    