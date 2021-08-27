# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:06:55 2021

@author: edi.piovezani
"""
import pandas as pd
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from xgboost                 import XGBClassifier
import time
from datetime import datetime


# Diretório padrão para leitura e gravação dos dados e modelos treinados
my_path = "/Users/edipiovezani/SKULD/"
#my_path = 'C:/Users/edi.piovezani/OneDrive - Omni S.A Crédito Financiamento e Investimento/Diretoria/10-Pessoal/IPT/SKULD/'
versao = '01'

# Define a estrutura da tabela result_table que armazenará a melhor configuração (hiperparâmetros) para cada classificador
result_table = pd.DataFrame(columns=['Classificador-Base', 
                                     'Classificador',
                                     'Base de Dados', 
                                     'Scale Metodo',
                                     'Alvo',
                                     'Tempo Processamento', 
                                     'Melhores Parametros', 
                                     'dt_inicio_processamento',
                                     'dt_fim_processamento',
                                     'Status_Processamento'
                                    ])

  
# Cria uma planilha Excel (CSV) no diretório definido para armazenar 
try:
    # Lê o arquivo (dataset) CSV e carrega na variável df (dataframe)
    result_table = pd.read_csv(my_path+'resultado_best_parameters_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
    result_table = result_table.iloc[:,1:]
    
except Exception:
    result_table.to_csv(my_path + 'resultado_best_parameters_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1')


#### Insere o resultado de cada classificador em uma tabela para facilitar a comparação
def insert_result_table (a,b,c,d,e,f,g,h,i,j):
    global result_table 
    result_table = result_table.append({'Classificador-Base':a,
                                        'Classificador':b,
                                        'Base de Dados':c,
                                        'Scale Metodo':d,
                                        'Alvo':e,
                                        'Tempo Processamento':f,
                                        'Melhores Parametros':g,
                                        'dt_inicio_processamento':h,
                                        'dt_fim_processamento':i,               
                                        'Status_Processamento':j
                                        }, ignore_index=True)

#### Executa todos os classificadores definidos na lista "classifiers" 
def executa_classificadores (nm_base, scale_method, X_train, y_train, alvo):   
    
    # Nomeia os classificadores 
    names = [
             'DecisionTree',
             'KNeighbors'  ,
             'LogisticRegression',
             'NeuralNetwork',
             'SVM',
             'XGBoost',
             'RandomForest',
             'AdaBoost',
             'GradientBoosting'
             ]
    
     # Dicionário de experimentação dos hiperparâmetros para encontrar a melhor configuração
    parameters = [
                   # DecisionTree 
                   [{'criterion':['gini','entropy'], 'max_depth':[5,10,30,50,100] }],    
                   # KNeighbors
                   [{'algorithm':['auto'], 'n_neighbors':[5,7,10,20], 'leaf_size':[3,5], 'n_jobs':[6]}],
                   # LogisticRegression
                   [{'solver':['lbfgs'], 'max_iter':[300,500,1000], 'C':[5,10,15], 'n_jobs':[7] }],
                   # MLPClassifier
                   [{'solver':['lbfgs'], 'max_iter':[500,800,1200], 'learning_rate_init':[0.001, 0.5, 1]}],
                   # SVM
                   [{'kernel':['linear','rbf','poly','sigmoid'], 'max_iter':[300, 500, 1000], 'probability':[True], 'cache_size':[4000]}],
                   #
                   ###### Ensemble métodos #########
                   # XGBClassifier
                   [{'booster':['gbtree'], 'eta':[0.1], 'max_depth':[5,10,30,50], 'n_estimators':[30, 50, 100, 150]}],
                   # RandomForest
                   [{'criterion':['gini','entropy'], 'max_depth':[10,30,50], 'n_estimators':[50, 100, 200, 500],'n_jobs':[7]}],
                   # AdaBoost
                   [{'algorithm':['SAMME.R'],'n_estimators':[300,500,800,1200], 'learning_rate':[0.01, 0.5, 1]}],
                   # GradientBoostingClassifier
                   [{'criterion':['friedman_mse','mse'], 'learning_rate':[0.01,0.1,0.3,0.5],'n_estimators':[800, 1000, 1200]}]
                 ]  
   
    # Instancia os objetos classificadores (modelos)
    classifiers = [
                   DecisionTreeClassifier(),
                   KNeighborsClassifier(),
                   LogisticRegression(),
                   MLPClassifier(),
                   SVC(),
                   XGBClassifier(),
                   RandomForestClassifier(),
                   AdaBoostClassifier(),
                   GradientBoostingClassifier()
                  ]
    
    # Treina e grava o resultado dos modelos
    for name, param, cls in zip(names, parameters, classifiers):
        # o IF abaixo verifica se o classificador já foi executado em algum processamento anterior para não processar novamente. 
        if name + '-' + nm_base + '-' + scale_method + '-' + alvo not in result_table.values:
            print(' ')
            print('Rodando o classificador: '+name + '-' + nm_base + '-' + scale_method + '-' + alvo + '...')
            data_e_hora_atual = datetime.now()
            data_e_hora_em_texto = data_e_hora_atual.strftime("%d/%m/%Y %H:%M")
            print('Inicio Processamento: '+ data_e_hora_em_texto)
            inicio = time.time()
            
            v_message   = 'OK'
            v_k_fold    = 5  # Quantidade de subconjuntos para fazer o cross-validation #
            cod_erro    = 0
            
            try:
                  # Executa o modelo/classificador/algoritmo recebido como parâmetro na variável cls
                  classificador = GridSearchCV(cls, param, scoring='accuracy', cv=v_k_fold, return_train_score=True, n_jobs=6)
                  classificador.fit(X_train, y_train)
                  modelo_final = classificador.best_estimator_  
                  
                  # Grava o modelo treinado em disco para posterior validação com a base de teste final
                  pickle.dump(modelo_final, open(my_path + name + '-' + nm_base + '-' + scale_method + '-' + alvo +'_V' + versao +'.sav','wb'))
                  
                  print(' ')
                  print('Base de Dados.: '+ nm_base + '\n' +
                        'Classificador.: '+ name + '-' + nm_base + '-' + scale_method + '-' + alvo)
                  print('Melhores Parametros: ' + str(modelo_final))
                  print(' ')
    
            except Exception as err:
               v_message = 'Erro: '+str(err)+' '+str(cod_erro)+'. Classificador: '+name + '-' + nm_base + '-' + scale_method + '-' + alvo
            
                     
            fim = time.time()
            tempo_processamento = (round((fim - inicio)/60,2))
            data_e_hora_atual = datetime.now()
            data_fim_processamento = data_e_hora_atual.strftime("%d/%m/%Y %H:%M")
            print('Tempo de processamento em Min/Seg: ' + str(int(tempo_processamento)) + ':%2.0f' % int((tempo_processamento - int(tempo_processamento)) * 60) +'\n')
            print(v_message+': '+name + '-' + nm_base + '-' + scale_method + '-' + alvo)
            ## Grava o resultado do processamento de cada classificador
            insert_result_table (name + '-' + nm_base + '-' + scale_method + '-' + alvo,
                                 name , 
                                 nm_base, 
                                 scale_method,
                                 alvo,
                                 tempo_processamento,
                                 str(modelo_final), 
                                 data_e_hora_em_texto, 
                                 data_fim_processamento,                                            
                                 v_message
                                )
                
            # Grava o resultado do processamento em um arquivo Excel csv 
            result_table.to_csv(my_path+'resultado_best_parameters_V'+versao+'.csv', sep=';',encoding = 'ISO-8859-1')

#####################################################################################################            
inicio_processamento = time.time()

scales_methods  = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler()]

# Carrega a lista de arquivos para o treinamento (carrega e converte para uma lista de strings)
bases_dados = pd.read_csv(my_path+'lista_bases_dados_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
bases_dados = list(bases_dados.iloc[:,1:].values.flatten())

# Carrega a lista de "alvos" para o treinamento
alvos = pd.read_csv(my_path+'alvos_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
alvos = list(alvos.iloc[:,1:].values.flatten())

############# Realiza o treinamento com todas as bases de dados da lista "base_dados" ###############
for bd in bases_dados:
    # Carrega o arquivo para treinamento 
    df_treinamento = pd.read_csv(my_path + bd , sep=';', encoding = 'ISO-8859-1', low_memory=False)
    df_treinamento = df_treinamento.iloc[:,1:]
        
    for x in alvos:
        variaveis   = df_treinamento.columns.values[:-len(alvos) * 3]
        base_treino = df_treinamento[variaveis].values
        alvo_treino = df_treinamento['ALVO' + str(x)].values
        
        #base_teste = out_of_sample[variaveis].values
        #alvo_teste = out_of_sample['ALVO' + str(x)].values
       
        #base_fora = out_of_time[variaveis].values
        #alvo_fora = out_of_time['ALVO' + str(x)].values
   
        executa_classificadores(bd, 'no_scale', base_treino, alvo_treino, str(x))        
############################  Fim da Execução do treinamento dos Modelos #############################        

fim_processamento = time.time()
tempo_processamento_geral = (round((fim_processamento - inicio_processamento)/60,2))
print("Tempo total de treinamento dos modelos em minutos: " ,str(int(tempo_processamento_geral)))





  


