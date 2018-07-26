#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v2.9


###############################################################################
## O estudo para a criação deste programa foi realizado  no arquivo:
## Deteccao_de_Fraude_Enron_Machine_Learning.ipynb
## que está neste mesmo diretório 
## 
## Abrir no jupyter notebook
###############################################################################


### Importando bibliotecas

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
import importlib
import tester

warnings.filterwarnings('always')

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, \ 
f_classif, RFECV

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn.tree import DecisionTreeClassifier#, export_graphviz

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


###############################################################################
### Carregando os dados

# Carregar o dicionário que contém o conjunto de dados
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
# Armazenar em my_dataset para facilitar a exportação a ser realizada no 
# final do arquivo
my_dataset = data_dict.copy()


# Para facilitar a análise dos dados, irei utilizar a biblioteca Pandas
df_data = pd.DataFrame.from_dict(my_dataset, orient='index')

# Vamos mudar substitiur os valores True  para 1 e False para 0
df_data['poi'] = df_data['poi'].map({True: 1,
                                     False: 0})
df_data = df_data.replace("NaN", np.nan)


###############################################################################
### Remoção de outliers

df_data.drop(['TOTAL'], inplace=True)
df_data.drop(['LOCKHART EUGENE E'], inplace=True)

my_dataset.pop('TOTAL')
my_dataset.pop('LOCKHART EUGENE E')



###############################################################################
### Conferência e correção de dados

df_data['total_payments'].fillna(0, inplace=True)
df_data['total_stock_value'].fillna(0, inplace=True)

# Realizado soma para conferência dos campos total_payments e total_stock_value
insider = 'BELFER ROBERT'
df_data['deferred_income'][insider] = df_data['deferral_payments'][insider]
df_data['deferral_payments'][insider] = .0
df_data['expenses'][insider] = df_data['director_fees'][insider]
df_data['director_fees'][insider] = df_data['total_payments'][insider]
df_data['total_payments'][insider] = df_data['exercised_stock_options'][insider]
df_data['exercised_stock_options'][insider] = df_data['restricted_stock'][insider]
df_data['restricted_stock'][insider] = df_data['restricted_stock_deferred'][insider]
df_data['restricted_stock_deferred'][insider] = df_data['total_stock_value'][insider]
df_data['total_stock_value'][insider] = .0

insider = 'BHATNAGAR SANJAY'
df_data['expenses'][insider]  = 137864.00 
df_data['other'][insider] = .0
df_data['total_payments'][insider] = 137864.00
df_data['director_fees'][insider] = .0
df_data['exercised_stock_options'][insider] = 15456290.00
df_data['restricted_stock'][insider] = 2604490.00
df_data['restricted_stock_deferred'][insider] = -2604490.00
df_data['total_stock_value'][insider] = 15456290.00


###############################################################################
## Criação de novas características

df_data['total_general'] = df_data['total_payments'] + df_data['total_stock_value']

total_payments_sum = sum(df_data['total_payments'])
total_stock_value_sum = sum(df_data['total_stock_value'])

if (total_payments_sum == np.nan) | (total_stock_value_sum == np.nan):
    print("Erro! Variável total_payments_sum ou total_stock_value_sum são nulas (nan)")
    exit()
    
   

for x in my_dataset:
    
    # total_general
    vtotal_payments = .0
    vtotal_stock_value = .0


    if my_dataset[x].get('total_payments') == 'NaN':
        vtotal_payments = .0
    else:
        vtotal_payments = vtotal_payments + float(my_dataset[x].get('total_payments'))
        
    if my_dataset[x].get('total_stock_value') == 'NaN':
        vtotal_stock_value = .0
    else:
        vtotal_stock_value = vtotal_stock_value + float(my_dataset[x].get('total_stock_value'))
        
    my_dataset[x]['total_general'] = vtotal_payments + vtotal_stock_value
    
    
    # perc_from_poi
    if my_dataset[x].get('to_messages') != 'NaN':
        if int(float(my_dataset[x].get('from_messages'))) > 0:
            my_dataset[x]['perc_from_poi'] = \
                    int(float(my_dataset[x].get('from_poi_to_this_person'))) / \
                    int(float(my_dataset[x].get('to_messages'))) 
        else:
            my_dataset[x]['perc_from_poi'] = 'NaN'
    else:
        my_dataset[x]['perc_from_poi'] = 'NaN'
        
        
    # perc_shared_poi
    if my_dataset[x].get('to_messages') != 'NaN':
    #if my_dataset[x].get('from_messages') != 'NaN':
        if int(float(my_dataset[x].get('from_messages'))) > 0:
            my_dataset[x]['perc_shared_poi'] = \
                    int(float(my_dataset[x].get('shared_receipt_with_poi'))) / \
                    int(float(my_dataset[x].get('to_messages'))) 
        else:
            my_dataset[x]['perc_shared_poi'] = 'NaN'
    else:
        my_dataset[x]['perc_shared_poi'] = 'NaN'
        
        
    # perc_to_poi
    if my_dataset[x].get('from_messages') != 'NaN':
        if int(float(my_dataset[x].get('from_messages'))) > 0:            
            my_dataset[x]['perc_to_poi'] = \
                    int(float(my_dataset[x].get('from_this_person_to_poi'))) / \
                    int(float(my_dataset[x].get('from_messages'))) 
        else:
            my_dataset[x]['perc_to_poi'] = 'NaN'
    else:
        my_dataset[x]['perc_to_poi'] = 'NaN'

        
    # perc_total_payments
    if my_dataset[x].get('total_payments') != 'NaN':
        my_dataset[x]['perc_total_payments'] = my_dataset[x].get('total_payments') / total_payments_sum
    else:
       my_dataset[x]['perc_total_payments'] = 'NaN'
    
    
    # perc_total_stock_value
    if my_dataset[x].get('total_stock_value') != 'NaN':
        my_dataset[x]['perc_total_stock_value'] = my_dataset[x].get('total_stock_value') / total_stock_value_sum
    else:
        my_dataset[x]['perc_total_stock_value'] = 'NaN'

        


### Selecionando as caracteristicas
features_list = ['poi',
'bonus', 
'deferral_payments',
'deferred_income',
'director_fees',
'exercised_stock_options',
'expenses', 
'from_messages',
'from_poi_to_this_person',
'from_this_person_to_poi',
'loan_advances',
'long_term_incentive',
'other',
'restricted_stock',
'restricted_stock_deferred',
'salary',
'shared_receipt_with_poi',
'to_messages',
'total_stock_value',
'total_payments']


features_list.append('total_general')
features_list.append('perc_from_poi')
features_list.append('perc_shared_poi')
features_list.append('perc_to_poi')
features_list.append('perc_total_payments')
features_list.append('perc_total_stock_value')



###############################################################################
### Separação dos dados

# Separar os dados de rótulo (lable) e das caracteristicas (features)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

X = features
y = labels

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,  random_state=42)

for train_index, test_index in sss.split(X, y):
    features_train = [X[ii] for ii in train_index]
    features_test  = [X[ii] for ii in test_index]
    labels_train = [y[ii] for ii in train_index] 
    labels_test  = [y[ii] for ii in test_index]  

    
    
###############################################################################
### Execução do algoritimo de Aprendizado de Máquina
    
CRITERION = ['gini','entropy']
SPLITTER = ['best', 'random']
MIN_SAMPLES_SPLIT = [3,5,10]
CLASS_WEIGHT = ['balanced', None]
MIN_SAMPLES_LEAF = [1,2,4,8,16]
MAX_DEPTH = [3, 5, 10]
SCALER = [StandardScaler(), MinMaxScaler(), None]
SELECTOR__K = [5, 8, 10, 15, 18] 
    
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('classifier', DecisionTreeClassifier(random_state=42))
])
    
param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'classifier__criterion': CRITERION,
    'classifier__splitter': SPLITTER,
    'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
    'classifier__class_weight': CLASS_WEIGHT,
    'classifier__min_samples_leaf': MIN_SAMPLES_LEAF,
    'classifier__max_depth': MAX_DEPTH,
}
    
grid_search = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
grid_search = grid_search.fit(features_train,labels_train)



###############################################################################
### Gravar arquivos para posterior validação pelo programa tester.py

model = grid_search.best_estimator_

dump_classifier_and_data(model, my_dataset, features_list)
