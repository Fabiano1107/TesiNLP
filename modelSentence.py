import numpy as np
from numpy import asarray
from numpy import zeros
from sklearn.metrics import accuracy_score
from keras import callbacks
from keras import optimizers
from keras.layers import *
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD

def importaWordSpace(pathWS = r'./ws.txt'):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(pathWS)
    for line in f:
        values = line.split()
        word = values[0]
        #print(word)
        string = ''.join(values[3:]).split(',')
        for x in string:
            x = float(x)
        #print(string)
        coefs = asarray(string, dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index

#embeddings_index = importaWordSpace()

# Creo la matrice con l'encoding dei documenti e ...
# Creo la matrice dei pesi per le parole dei documenti di training
def creoMatriceFrasi(sentences):
    np.random.seed(100)
    # Eseguo il tokenizer
    t = Tokenizer()
    t.fit_on_texts(sentences)
    # Trasformo le frasi in sequenze di documenti
    encoded_docs = t.texts_to_sequences(sentences)
    #print(encoded_docs)
    # Eseguo un pad di dimensione 15
    max_length = 15
    # data è la matrice delle frasi
    data = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return data

# Creo una matrice pesata per le parole nei documenti
def creoMatriceVoc(sentences,embeddings_index):
    np.random.seed(100)
    # Eseguo il tokenizer
    t = Tokenizer()
    t.fit_on_texts(sentences)
    # Trasformo le frasi in sequenze di documenti
    encoded_docs = t.texts_to_sequences(sentences)
    #print(encoded_docs)
    # Eseguo un pad di dimensione 15
    max_length = 15
    # data è la matrice delle frasi
    data = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    vocab_size = len(t.word_index) + 1
    #print("vocab_size", vocab_size)
    sigma, mu = 0.5, 0
    embedding_matrix = sigma * np.random.randn(vocab_size, 250) + mu
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, vocab_size

### Definisco il modello: architettura NN
def getLstm(embedding_matrix,vocab_size):
    
    model = Sequential()
    model.add(Embedding(vocab_size, 250, weights=[embedding_matrix], input_length=15, trainable=True))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(8, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-6)
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    #
    # creo anche un modello sequenziale da provare individualmente
    #
    
    syn_input = Input(shape=(15,))
    x = (Embedding(vocab_size, 250, weights=[embedding_matrix], trainable=True))(syn_input)
    y = (Bidirectional(LSTM(1000, dropout=0.2, recurrent_dropout=0.2)))(x)
    syntax = (Dense(3000,activation='sigmoid'))(x)
    
    return syn_input, syntax, model

def addestroSentences(model,x_train, y_train):
    ## Fit the model
    #ea = callbacks.EarlyStopping(patience=5), callbacks = [ea]
    model.fit(x_train, y_train, epochs=50, batch_size=100)