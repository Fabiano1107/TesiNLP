import spacy
from nltk import Tree

from pydtk.tree import Tree as dtkTree
from pydtk.dtk import DT
from pydtk.operation import fast_shuffled_convolution

import re
import string


def clean_text(text):

    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    # stops = set(stopwords.words("italian"))
    # text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " , ", text) 
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

########################################################################################################################
#########################__________SPACY NLTK______________________________#############################################
########################################################################################################################

def getDoc(frase):

    ## ritorna il doc della frase parsata

    # fa il parse della frase in italiano
    nlp = spacy.load('it')

    # pulisce la frase
    frase = clean_text(frase)

    doc = nlp(frase)

    return doc


def printTree(doc):

    ## ritorna l'albero delle dipendenze del doc della frase parsata e lo stampa

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
        else:
            return node.orth_

    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

    print([to_nltk_tree(sent.root) for sent in doc.sents])


def getSentence(doc):
    
    ## preso in input un albero delle dipendenze,
    ## converte questo in un albero dei costituenti 

    dizionario = doc.print_tree()
    global stringa
    stringa= '('

    def getSentenceRicorsiva(dizionario):
        global stringa
        for elem in dizionario:
            stringa = stringa + elem['POS_coarse'] + ' (' + elem['word']
            if elem['modifiers'] != []:
                stringa = stringa + ' ('
                getSentenceRicorsiva(elem['modifiers'])
                stringa = stringa + ')'
            stringa = stringa + ') '

    getSentenceRicorsiva(dizionario)
    stringa = stringa + ')'
    stringa = re.sub(r"\)+\s+\)", "))", stringa) 
    return stringa



########################################################################################################################
#########################_______________PYDTK______________________________#############################################
########################################################################################################################
def pydtk(stringa):
    tree = dtkTree(string=stringa)
    dtCalculator = DT(dimension=3000, LAMBDA= 0.6, operation=fast_shuffled_convolution)

    distributedTree = dtCalculator.dt(tree)

    return distributedTree

# lista = df['SEN'][1:]
def getVector(lista):
    ## ritorna la lista dei vettori
    
    lista_vettori = []
    for frase in lista:
        doc = getDoc(frase)
        stringa = getSentence(doc)
        vettore = pydtk(stringa)
        lista.append(vettore)
    
    return lista_vettori