# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 21:44:25 2021
@author: edi.piovezani
"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime               import datetime
import time

# Diretório padrão para leitura e gravação dos dados e modelos treinados
my_path = "/Users/edipiovezani/SKULD/"
# my_path = 'C:/Users/edi.piovezani/OneDrive - Omni S.A Crédito Financiamento e Investimento/Diretoria/10-Pessoal/IPT/SKULD/'
versao = '01'
            
# Define a estrutura da tabela result_table que armazenará
# o resultado de todos os modelos/classificadores
result_table = pd.DataFrame(columns=['Classificador-Base', 
                                     'Scale Metodo',
                                     'Arquivo Teste',
                                     'Accuracy', 
                                     'Precision', 
                                     'Recall', 
                                     'Specificity', 
                                     'G-mean',
                                     'F1 Score', 
                                     'AUC', 
                                     'Average Precision',
                                     'Tempo Processamento', 
                                     'dt_inicio_processamento',
                                     'dt_fim_processamento',
                                     'Status_Processamento'                                     
                                    ])

#### Insere o resultado de cada classificador em uma tabela para facilitar a comparação
def insert_result_table (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):
    print(a)
    global result_table 
    result_table = result_table.append({'Classificador-Base':a,
                                        'Scale Metodo': b,
                                        'Arquivo Teste':c,
                                        'Accuracy':d,
                                        'Precision':e,
                                        'Recall':f,
                                        'Specificity':g,
                                        'G-mean':h,
                                        'F1 Score':i,
                                        'AUC':j,
                                        'Average Precision':k,
                                        'Tempo Processamento':l,
                                        'dt_inicio_processamento':m,
                                        'dt_fim_processamento':n,               
                                        'Status_Processamento':o
                                        }, ignore_index=True)

inicio_processamento = time.time()

# Cria um arquivo CSV para armazenar o resultado dos modelos ou, caso já exista um arquivo com informações parciais, 
# utiliza o existente / incluí novas informaçõe no mesmo arquivo
try:
    result_table = pd.read_csv(my_path+'resultado_dos_modelos_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
    result_table = result_table.iloc[:,1:]
except Exception:
    result_table.to_csv(my_path + 'resultado_dos_modelos_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1')


# Lê o arquivo com todos os classificadores com seus melhores hipermarÂmetros
best_parameter = pd.read_csv(my_path+'resultado_best_parameters_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
best_parameter = best_parameter.iloc[:,1:]


# Carrega a lista de "alvos" para o treinamento
alvos = pd.read_csv(my_path+'alvos_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
alvos = list(alvos.iloc[:,1:].values.flatten())

## Cria a lista dos Classificadores que serão testados 
classifiers    = best_parameter['Classificador-Base'] 
scaler         = best_parameter['Scale Metodo'] 
alvo           = best_parameter['Alvo']
status         = best_parameter['Status_Processamento']
arquivos_teste = ['df_teste_','df_teste_out_of_time_']
   
#### Testa todos os classificadores definidos na lista "classifiers"
# carrega os modelos previamente treinados, testa e calcula os indicadores de desempenho
for v_cls, v_scaler, v_alvo, v_status in zip(classifiers, scaler, alvo, status):
    # o IF abaixo verifica se o classificador já foi executado em algum processamento anterior para não processar novamente. 
    if v_cls not in result_table.values and v_status == 'OK':
        acuracia, precisao, recall, f1, specificity, gmean, auc, fpr, tpr = 0, 0, 0, 0, 0, 0, 0, 0, 0
        print(' ')
        print('Testando o classificador: '+ v_cls)
        data_e_hora_atual = datetime.now()
        data_e_hora_em_texto = data_e_hora_atual.strftime("%d/%m/%Y %H:%M")
        print('Inicio Processamento: '+ data_e_hora_em_texto)
        inicio = time.time()
        v_message   = 'OK'
        cod_erro    = 0
                   
        for tipo_arquivo_teste in arquivos_teste:
            print(tipo_arquivo_teste)
         
            try:            
                # Encontra o nome do ativo na string maior com o nome completo do modelo
                ativo = v_cls[v_cls.index('.')-5: v_cls.index('.')+3]
                
                # Carrega o arquivo de teste 
                if tipo_arquivo_teste == 'df_teste_':
                    v_nome_arquivo_teste = tipo_arquivo_teste + ativo + '_V'+versao
                    df_teste = pd.read_csv(my_path + v_nome_arquivo_teste +'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
                    df_teste = df_teste.iloc[:,1:]
                if tipo_arquivo_teste == 'df_teste_out_of_time_':
                    v_nome_arquivo_teste = tipo_arquivo_teste + ativo + '_V'+versao
                    df_teste = pd.read_csv(my_path + v_nome_arquivo_teste +'.csv', sep=';', encoding = 'ISO-8859-1', low_memory=False)
                    df_teste = df_teste.iloc[:,1:]
                          
                variaveis  = df_teste.columns.values[:-len(alvos) * 3]
                base_teste = df_teste[variaveis].values
                alvo_teste = df_teste['ALVO' + str(v_alvo)].values
                            
                # Carrega o modelo que está gravado no diretorio e faz o teste
                modelo      = pickle.load(open(my_path+v_cls+'_V'+versao+'.sav','rb'))
                ypredict    = modelo.predict (base_teste) 
                yproba      = modelo.predict_proba(base_teste)[::,1]
                    
                # Calcula Matriz de Confusão #
                np.set_printoptions(precision=2)
                tn, fp, fn, tp = confusion_matrix(alvo_teste, ypredict).ravel()
                cm = np.array([[tn, fp], [fn,tp]], np.int32)
                            
                print(cm)
                print(' ')
                print('TN: '+str(tn)+', FP: '+str(fp)+', FN: '+str(fn)+', TP: '+str(tp)+'.')
                print(' ')
               
                # Calcula as métricas de desempenho dos modelos/classificadores  
                acuracia    = (tp+tn) / (tp+tn+fp+fn)   
                precisao    = tp / (tp+fp)              
                recall      = tp / (tp+fn)              
                f1          = 2 * ((precisao*recall)/(precisao+recall))  
                specificity = tn / (tn+fp)              
                gmean       = math.sqrt(recall * specificity)
         
                ###### Inicio Plotagem da Matriz de Confusão #########################
                plt.clf() # Limpa a figura corrente #
                plt.figure(figsize=(6.75,4.5))
                classNames = ['N=Não', 'S=Sim']
                plt.title('Matriz Confusao do Modelo'+'\n'+ v_cls + ' V' + versao + '\n' + v_nome_arquivo_teste, fontweight='bold', fontsize=10,  horizontalalignment="center")
                plt.ylabel('Rótulos Reais/Verdadeiros', fontsize=9)
                plt.tight_layout()
                plt.xlabel('Rótulos Previstos pelo Modelo',fontsize=9)
                tick_marks = np.arange(len(classNames))
                plt.xticks(tick_marks, classNames, fontsize=9, horizontalalignment="center", rotation=0)
                plt.yticks(tick_marks, classNames, fontsize=9)
                s = [['TN','FP'], ['FN', 'TP']]
                thresh = cm.max() / 2.
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]),fontsize=10, fontweight='bold', color="white" if cm[i,j] > thresh else "black",  horizontalalignment="center")
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.savefig ('Matriz_Confusao_'+ v_cls + '_V'+versao+'_'+v_nome_arquivo_teste+'.png', format='png')
                plt.show()
                ###### Fim da Plotagem da Matriz de Confusão #########################       
                              
                ###### Matriz de Confusão - Outro layout #############################
                display = ConfusionMatrixDisplay(confusion_matrix=cm/len(alvo_teste), display_labels=modelo.classes_)
                display.plot(values_format= '0.2%')
                display.ax_.set_title('Matriz Confusao do Modelo'+'\n'+ v_cls + ' V' + versao + '\n' + v_nome_arquivo_teste, fontweight='bold', fontsize=10,  horizontalalignment="center")
                #######################################################################               
                
                print(' ')
                print('Base de Dados.: '+ v_cls + ' V' + versao)
                print('Accuracy .....: %0.2f ' % (acuracia) )
                print('Precision ....: %0.2f ' % (precisao) )
                print('Recall .......: %0.2f ' % (recall  ) )
                print('Specificity...: %0.2f ' % (specificity) )
                print('G-mean........: %0.2f ' % (gmean) )
                print('F1 Score .....: %0.2f ' % (f1) )
                print('AUC...........: %0.2f ' % (auc) )
                print(' ')
    
            except Exception as err:
               v_message = 'Erro: '+str(err)+' '+str(cod_erro)+'. Classificador: '+ v_cls            
                     
            fim = time.time()
            tempo_processamento = (round((fim - inicio)/60,2))
            data_e_hora_atual = datetime.now()
            data_fim_processamento = data_e_hora_atual.strftime("%d/%m/%Y %H:%M")
            print('Tempo de processamento em Min/Seg: ' + str(int(tempo_processamento)) + ':%2.0f' % int((tempo_processamento - int(tempo_processamento)) * 60) +'\n')
            print(v_message+': '+v_cls)
            print('===============================================================')
          
            ## Grava o resultado do processamento de cada classificador
            insert_result_table (v_cls,
                                 v_scaler,
                                 v_nome_arquivo_teste,
                                 acuracia, 
                                 precisao, 
                                 recall,
                                 specificity,
                                 gmean,
                                 f1,
                                 auc, 
                                 0,
                                 tempo_processamento,
                                 data_e_hora_em_texto, 
                                 data_fim_processamento,                                            
                                 v_message
                                )
                
            # Grava o resultado do processamento em um arquivo Excel csv 
            result_table.to_csv(my_path+'resultado_dos_modelos_V'+versao+'.csv', sep=';',encoding = 'ISO-8859-1')

####################### Fim dos Testes dos modelos ###############################


fim_processamento = time.time()
tempo_processamento_geral = (round((fim_processamento - inicio_processamento)/60,2))
print("Tempo total de processamento em minutos: " ,str(int(tempo_processamento_geral)))

