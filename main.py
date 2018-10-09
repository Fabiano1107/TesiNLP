from readDataset import *
from parserAlbero import *
from modelTree import *
from modelSentence import *
from mlp import *
import numpy as np
import pickle

df = getDataset()

# inserisco in sentences le frasi pulite
sentences = df['SEN'].map(lambda x: clean_text(x))[1:] # elimino la prima riga perchè contiene l'intestazione

# estraggo i target del dataset
target = getTarget(df)

# dal dataset, estraggo separatamente i training set e i test set
x_train, x_test, y_train, y_test = splitData(df['SEN'][1:],target)

############
### Parte relativa alle sentences
############
#print(sentences)

embeddings_index = importaWordSpace()
x_train_sen = creoMatriceFrasi(x_train)
x_test_sen = creoMatriceFrasi(x_test)
embedding_matrix, vocab_size = creoMatriceVoc(sentences,embeddings_index)

sem_input, semantics, modelTree = getModelTree()
syn_input, syntax, modelLstm = getLstm(embedding_matrix,vocab_size)
final_model = getMlp(semantics,syntax,sem_input,syn_input)

x_train_vec = getVector(x_train.tolist())
x_test_vec = getVector(x_test.tolist())

final_model.fit([np.asarray(x_train_vec),x_train_sen], y_train, epochs=250, batch_size=300,class_weight=[0.999,0.001])

predict(final_model,[np.asarray(x_train_vec),np.asarray(x_train_sen)],[np.asarray(x_test_vec),np.asarray(x_test_sen)],y_train,y_test)

## train modello lstm
#addestroSentences(modelLstm,x_train_sen,y_train)
#predict(modelLstm,x_train_sen,x_test_sen,y_train,y_test)

## train modello albero
#addestraTree(modelTree,np.asarray(x_train_vec),np.asarray(y_train))
#predict(modelTree,np.asarray(x_train_vec),np.asarray(x_test_vec),np.asarray(y_train),np.asarray(y_test))