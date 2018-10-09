from modelTree import *
from modelSentence import *
import keras.optimizers as opt

def getMlp(model1,model2):
    final_model = Sequential()
    final_model.add(Concatenate([model1,model2]))
    final_model.add(LSTM(300))
    final_model.add(Dense(output_dim=8))
    final_model.add(Activation("relu"))
    final_model.add(Activation("softmax"))
    optim = opt.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    final_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return final_model

def addestraMlp(model,x_train_vec,x_train_sec,y_train):
    final_model.fit([x_train_vec,x_train_sen], y_train, nb_epoch=1, batch_size=16,class_weight=[0.999,0.001])
