from keras import models,layers,losses,optimizers,metrics,callbacks
from tensorflow_addons.losses import sigmoid_focal_crossentropy
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC,LinearSVC
import numpy as np
class MyNetWork:
    def augmenting(self,X:np.ndarray):
        return np.hstack((X,np.sqrt(X),np.log2(X*0.75+1),np.exp2(X),X**2))

    def __init__(self) -> None:
        pass

    def fit(self,X,Y):
        X=self.augmenting(X)
        N=len(X[0])
        xin=layers.Input(shape=(N,))
        x=layers.Dense(N)(xin)
        x=layers.BatchNormalization()(x)
        x=layers.Add()([x,xin])
        x=layers.Dense(1)(x)
        x=layers.Activation('sigmoid')(x)
        self.model=models.Model(inputs=xin,outputs=x)
        self.model.compile(
            loss=sigmoid_focal_crossentropy,
            optimizer=optimizers.Adam()
        )
        cb = callbacks.EarlyStopping(monitor='loss', patience=6)
        return self.model.fit(X,Y,validation_split=0.3,epochs=100,batch_size=64,callbacks=[cb])
    
    def predict(self,X):
        X=self.augmenting(X)
        return self.model.predict(X,batch_size=4096)

class MyLinearSVC:
    def augmenting(self,X:np.ndarray):
        def cross(X:np.ndarray):
            temp=[X]
            for i in range(X.shape[1]):
                for j in range(i,X.shape[1]):
                    temp.append(X[:,i:i+1]*X[:,j:j+1])
            return np.hstack(temp)
        return cross(np.hstack((X,np.log2(0.75*X+1))))

    def __init__(self) -> None:
        self.model=LinearSVC(C=0.5)
    def fit(self,X,Y):
        X=self.augmenting(X)
        self.model.fit(X,Y)
    def predict(self,X):
        X=self.augmenting(X)
        return self.model.decision_function(X)

class MyRidgeClassifier:
    def augmenting(self,X:np.ndarray):
        def cross(X:np.ndarray):
            temp=[X]
            for i in range(X.shape[1]):
                for j in range(i,X.shape[1]):
                    temp.append(X[:,i:i+1]*X[:,j:j+1])
            return np.hstack(temp)
        return cross(np.hstack((X,np.log2(0.75*X+1))))

    def __init__(self) -> None:
        self.model=RidgeClassifier(alpha=0.9,positive=True,solver='lbfgs')
    def fit(self,X,Y):
        X=self.augmenting(X)
        self.model.fit(X,Y)
    def predict(self,X):
        X=self.augmenting(X)
        return self.model.decision_function(X)