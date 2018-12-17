import numpy as np 
import pandas as pd 
import nltk

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Leitura dos dados de treino e do dataset a ser classificado
#em pandas dataframes
train=pd.read_csv('training_set.tsv',sep='\t')
test=pd.read_csv('data_estag_ds.tsv',sep='\t')

# Atribuição de valores para os rótulos
train['NCLASS']=train['CLASS'].apply({'smartphone':0,  'não-smartphone':1}.get)

##########################################################################################
# Pré-processamento dos conjuntos de dados
X_text_train=train['TITLE'].values
X_text_test=test['TITLE'].values
y=train['NCLASS'].values

stop_words = set(nltk.corpus.stopwords.words('portuguese'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '-'])
stemmer = nltk.stem.RSLPStemmer()

processed_train = []
for doc in X_text_train:
    tokens = nltk.tokenize.word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_train.append(stemmed)

processed_test = []
for doc in X_text_test:
    tokens = nltk.tokenize.word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_test.append(stemmed)

train['processed_train']=processed_train
test['processed_test']=processed_test

row_lst = []
for lst in train.loc[:,'processed_train']:
    text = ''
    for word in lst:
        text = text + ' ' + word
    row_lst.append(text)

train['final_processed_text'] = row_lst

row_lst = []
for lst in test.loc[:,'processed_test']:
    text = ''
    for word in lst:
        text = text + ' ' + word
    row_lst.append(text)

test['final_processed_test'] = row_lst
#########################################################################################

#aplicação do tf-idf nos dados 
tfidf = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'))
tfidf.fit(train['final_processed_text'])

X_train_tfidf = tfidf.transform(train['final_processed_text'])
X_test_tfidf = tfidf.transform(test['final_processed_test'])

#criação do modelo para classificação
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y)

#predição dos dados do conjunto
predictions = rf.predict(X_test_tfidf)


#inserção dos rótulos preditos na tabela de resultado
test['NCLASS']=predictions

test['CLASS']=test['NCLASS'].apply({0:'smartphone',  1:'não-smartphone'}.get)

test = test.drop(columns=['processed_test', 'final_processed_test', 'NCLASS'], axis=1)
test.to_csv('ofertas_classificadas.tsv',sep='\t')