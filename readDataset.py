import numpy as np
import pandas as pd

def getDataset():
    df = pd.read_csv(r'C:/Users/Fabiano/Anaconda3/envs/NLP/NLP/absita_2018_training.csv', header=None, sep=';', encoding = 'utf8')
    df.columns = ['ID','CL','CLP','CLN','COM','COMP','COMN','AME','AMEP','AMEN','STA','STAP','STAN','VAL','VALP','VALN','WIF','WIFP','WIFN','LOC','LOCP','LOCN','OTH','OTHP','OTHN','SEN']
    df.head()

    return df

def check(df):
    errori = 0
    for colonna in  range(1,len(df.columns)-2,3):
        esiste = df.columns[colonna]
        positivo = df.columns[colonna + 1]
        negativo = df.columns[colonna + 2]
        for count in range(1,len(df[esiste])):

            if df[esiste][count] is not '0' and df[esiste][count] is not '1':
                print ('valore ' + esiste + ' diverso da 1 o 0')
                print (df[esiste][count])
                print (esiste,count)
                print ('\n')
                errori +=1
                #return False
            
            if df[positivo][count] is not '0' and df[positivo][count] is not '1':
                print ('valore' + positivo + 'diverso da 1 o 0')
                print (df[positivo][count])
                print (positivo,count)
                print ('\n')
                errori +=1
                #return False
            
            if df[negativo][count] is not '0' and df[negativo][count] is not '1':
                print ('valore' + negativo + 'diverso da 1 o 0')
                print (df[negativo][count])
                print (negativo,count)
                print ('\n')
                errori +=1
                #return False

            if df[esiste][count] is '1':     #se esiste == 1, uno dei due deve essere 0
                if df[positivo][count] is '0' and df[negativo][count] is '0': 
                    print ('Errore con presenza = 1 ed entrambi 0 o 1')
                    print (df[esiste][count],df[positivo][count],df[negativo][count])
                    print (esiste,positivo,negativo,count)
                    print (df['SEN'][count])
                    print ('\n')
                    errori +=1
                    #return False
            
            if df[esiste][count] is '0':     #se esiste == 0, anche gli altri devono essere 0
                if df[positivo][count] is not '0' or df[negativo][count] is not '0':
                    print ('Errore con presenza = 0 e valori 1 a destra')
                    print (df[esiste][count],df[positivo][count],df[negativo][count])
                    print (esiste,positivo,negativo,count)
                    print (df['SEN'][count])
                    print ('\n')
                    errori +=1
                    #return False

    print('Sono presenti ' + str(errori) + ' errori')


def getTarget(df):

    # Se si parla di una certa feature o meno!
    prediction_columns = ['CL','COM','AME','STA','VAL','WIF','LOC','OTH']
    target = df[prediction_columns][1:]

    return target
