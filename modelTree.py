from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Concatenate, concatenate
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np


def getModelTree():
    sem_input = Input(shape=(3000,))
    semantics = Dropout(0.2)(Dense(6000, activation='sigmoid')(Dropout(0.2)(Dense(3000,activation='relu')(sem_input))))

    # primo dens che prende in input un vettore di dimensione 3000
    # e restituisce un vettore piú grande di dimensione 6000
    model = Sequential()
    model.add(Dense(6000, input_dim=3000, activation='relu'))
    model.add(Dropout(0.1))
    # ritorna 8 features
    model.add(Dense(8, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    #
    # creo anche un modello sequenziale da provare individualmente
    #
    return sem_input,semantics,model

def splitData(data,target):
    # Divido il dataset in trainig e test set
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.2, random_state=0)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    return x_train, x_test, y_train, y_test

# pulisce i dati predetti
def afterPredict(predict):
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if predict[i][j] > 0.5:
                predict[i][j] = 1
            else:
                predict[i][j] = 0
    return predict

def predict(model,x_train, x_test, y_train, y_test):

    predict_train = model.predict(x_train)
    predict_train = afterPredict(predict_train)
    predict_train = predict_train.astype(int)
    print("Accuracy Trainig:\t", accuracy_score(y_train, predict_train))
    
    predict_test = model.predict(x_test)
    predict_test = afterPredict(predict_test)
    predict_test = predict_test.astype(int)
    print("Accuracy Test:\t\t", accuracy_score(y_test, predict_test))

def addestraTree(model,x_train,y_train):
    model.fit(x_train, y_train,epochs=5,batch_size=128)
