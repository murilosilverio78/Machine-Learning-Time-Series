# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 23:37:54 2021
"""
import yfinance as yf
import pandas as pd
import talib  as ta
from sklearn.model_selection import train_test_split

# Diretório padrão para leitura e gravação dos dados e modelos treinados
#my_path = "/Users/edipiovezani/SKULD/"
my_path = "/Users/muril/SKULD/"
# my_path = 'C:/Users/edi.piovezani/OneDrive - Omni S.A Crédito Financiamento e Investimento/Diretoria/10-Pessoal/IPT/SKULD/'
versao = '01'

# Função que carrega os dodos financeiros de um ativo entre as datas especificadas
def carrega_dados(ativo, data_inicio, data_fim):
    df = yf.download(ativo, data_inicio, data_fim) 
    return df

# Função que cria n janelas de dados baseado no deslocamento de um dia. 
# Ex: Se o parâmetro periodo for igual a 5, para cada dia
# de dado de um ativo serão gerados mais 5 conjuntos de dados com a visão D-1, D-2 ...D-5
def desloca_periodo(dataset, periodo):
    ac = pd.DataFrame()
    ac['OPEN_T' + str(periodo)]     = dataset['Open'].shift(-periodo)
    ac['HIGH_T' + str(periodo)]     = dataset['High'].shift(-periodo)
    ac['LOW_T' + str(periodo)]      = dataset['Low'].shift(-periodo)
    ac['CLOSE_T' + str(periodo)]    = dataset['Close'].shift(-periodo)
    ac['ADJCLOSE_T' + str(periodo)] = dataset['Adj Close'].shift(-periodo)
    ac['VOLUME_T' + str(periodo)]   = dataset['Volume'].shift(-periodo)
    
    dataset['OPEN_T' + str(periodo)]     = ac['OPEN_T' + str(periodo)]
    dataset['HIGH_T' + str(periodo)]     = ac['HIGH_T' + str(periodo)]
    dataset['LOW_T' + str(periodo)]      = ac['LOW_T' + str(periodo)]
    dataset['CLOSE_T' + str(periodo)]    = ac['CLOSE_T' + str(periodo)]
    dataset['ADJCLOSE_T' + str(periodo)] = ac['ADJCLOSE_T' + str(periodo)]
    dataset['VOLUME_T' + str(periodo)]   = ac['VOLUME_T' + str(periodo)]
    
    return dataset


# Função que cria os alvos a serem previstos.  Retorna a previsão de retorno após x dias e também um booleano baseado no 
# parâmetro limite, onde caso o retorno seja superior a esse limite, é retornado S, caso contrário N.  O parâmetro período
# indica qual a janela de tempo a ser prevista. Deve ser um array de valores.
def cria_alvo(dataset, periodo, limite):
    ac = pd.DataFrame()
    ac['ADJCLOSE_T' + str(periodo)]       = dataset['Adj Close'].shift(-periodo)
    dataset['ADJCLOSE_T' + str(periodo)]  = ac['ADJCLOSE_T' + str(periodo)]
    dataset['PERFORM' + str(periodo)]     = ((dataset['ADJCLOSE_T' + str(periodo)] / dataset['Adj Close']) ** (21 / periodo)) - 1
    dataset['ALVO' + str(periodo)]        = 'N'
    dataset['ALVO' + str(periodo)]        = dataset['ALVO' + str(periodo)].where(dataset['PERFORM' + str(periodo)] < limite, 'S')
                                        
    return dataset


# Função que cria indicadores técnicos para o ativo estudado
def indicadores_tecnicos(dataset, parametros):

    dataset['RSI']         = ta.RSI(dataset['Adj Close'], 
                                    timeperiod = parametros['rsi_timeperiod'])

    dataset['WILLR']       = ta.WILLR(dataset['High'], dataset['Low'], dataset['Adj Close'], 
                                    timeperiod = parametros['willr_timeperiod'])
 
    dataset['MACD'],       \
    dataset['MACDSIGNAL'], \
    dataset['MACDHIST']    = ta.MACD(dataset['Adj Close'], 
                                    fastperiod = parametros['macd_fastperiod'], 
                                    slowperiod = parametros['macd_slowperiod'], 
                                    signalperiod = parametros['macd_signalperiod'])

    dataset['OBV']         = ta.OBV(dataset['Adj Close'], dataset['Volume'])

    dataset['ROC']         = ta.ROC(dataset['Adj Close'], 
                                    timeperiod = parametros['roc_timperiod'])

    dataset['FASTK'],      \
    dataset['FASTD']       = ta.STOCHRSI(dataset['Adj Close'], 
                                    timeperiod=parametros['fast_timeperiod'], 
                                    fastk_period=parametros['fastk_period'], 
                                    fastd_period=parametros['fastd_period'], 
                                    fastd_matype=parametros['fastd_matype'])

    return dataset


# Função que gera três bases: Uma base de treinamento (sample), outra de teste (out_of_sample) e outra com dados posteriores aos
# dados utilizados no treinamento do modelo.  A base sample e out_of_sample contém dados da mesma janela de tempo.
def gera_base(ativo, data_inicio, data_fim, dias_dados, it_parametros, alvos, limite):

    dataset = carrega_dados(ativo, data_inicio, data_fim)
    
    # Cria o dia da semana, do mês, o mês e o ano como variáveis independentes
    dataset['DIA_SEMANA']   = dataset.index.dayofweek
    dataset['DIA_MES']      = dataset.index.day
    dataset['MES']          = dataset.index.month
    dataset['ANO']          = dataset.index.year

    for x in range(dias_dados):
        desloca_periodo(dataset, - (x + 1))

    dataset = indicadores_tecnicos(dataset, it_parametros)

    for x in alvos:
        dataset = cria_alvo(dataset, x, limite)

    # Renomeia as colunas do DataFrame
    dataset.rename(columns={"Open":"OPEN_T0", 
                            "High":"HIGH_T0", 
                            "Low":"LOW_T0", 
                            "Close":"CLOSE_T0", 
                            "Adj Close": "ADJCLOSE_T0", 
                            "Volume":"VOLUME_T0"}, inplace = True)
    
    # Ordena o DataFrame por ordem cronologica, do dado mais novo para o mais antigo
    dataset.sort_values(by=['Date'], ascending=False, inplace = True)
    
    # Elimina as linhas com campos nulos.  Esses campos nulos são gerados pelas janelas de tempo necessárias para criar os 
    # indicadores técnicos e também os alvos
    dataset = dataset.dropna()
    
    # Separa 15% da base como base fora do tempo dos dados utilizados para treinamento do modelo
    out_of_time = dataset.iloc[:int(round(len(dataset.index) * 0.15, 0)),:]
    dataset = dataset.iloc[int(round(len(dataset.index) * 0.15, 0)):,:]
    
    # Separa o restante da base em base de desenvolvimento e base de teste, 85% e 15% do tamanho da base, respectivamente 
    sample, out_of_sample = train_test_split(dataset, test_size=0.15)
    
    return sample, out_of_sample, out_of_time


# Parâmetros necessários para executar as funções
data_inicio = "2015-02-11"
data_fim = "2020-02-10"
it_parametros = {
    "rsi_timeperiod": 14,
    "willr_timeperiod": 14,
    "macd_fastperiod": 12,
    "macd_slowperiod": 26,
    "macd_signalperiod": 9,
    "roc_timperiod": 10,
    "fast_timeperiod": 14,
    "fastk_period": 5,
    "fastd_period": 3,
    "fastd_matype": 0}
dias_dados = 15
limite = 0.01
alvos = [5, 21,30]
df_alvos = pd.DataFrame(alvos)     
df_alvos.to_csv (my_path + 'alvos_V'+versao+'.csv', sep=';', encoding = 'ISO-8859-1')

# Lista dos ativos
ativos = ['VALE3.SA','PETR3.SA','ITSA4.SA','JBSS3.SA']

# Define uma tabelinha para guardar o nome dos arquivos (bases de dados) que serão gerados
global lista_bases_dados 
lista_bases_dados = pd.DataFrame(columns=['Base de Dados'])
  
# Gera as bases de dados para serem utilizadas na etapa de Treinamento
for ticker in ativos:
    # Chama a função para criar as bases
    sample, out_of_sample, out_of_time = gera_base(ticker, data_inicio, data_fim, dias_dados, it_parametros, alvos, limite)
    
    # Grava as bases em arquivo CSV para serem utilizadas na etapa de treinamento 
    lista_bases_dados = lista_bases_dados.append({'Base de Dados':'df_trein_'+ticker+'_V'+versao+'.csv'}, ignore_index=True)
    sample.to_csv         (my_path + 'df_trein_'+ticker+'_V'+versao+'.csv'               , float_format='%.3f', sep=';', encoding = 'ISO-8859-1')
    out_of_sample.to_csv  (my_path + 'df_teste_'+ticker+'_V'+versao+'.csv'               , float_format='%.3f', sep=';', encoding = 'ISO-8859-1')
    out_of_time.to_csv    (my_path + 'df_teste_out_of_time_'+ticker+'_V'+versao+'.csv'   , float_format='%.3f', sep=';', encoding = 'ISO-8859-1')
    
    ###  Verifica o balanceamento da base para treinamento dos modelos #####
    for alvo in alvos:
        target_count = sample['ALVO' + str(alvo)].value_counts()
        qtde_registro = target_count['S'] + target_count['N']
        target_count.plot(kind='bar', title='Contagem da variável Alvo'+str(alvo),color = ['#1F77B4', '#FF7F0E'])
        print('Sim......:', target_count['S'])
        print('Não......:', target_count['N'])
        print('Proporção SIM:', round((target_count['S'] / qtde_registro)*100,2),'%')
        print('Proporção NÃO:', round((target_count['N'] / qtde_registro)*100,2),'%')


# Grava o resultado do processamento em um arquivo Excel csv 
lista_bases_dados.to_csv(my_path+'lista_bases_dados_V'+versao+'.csv', sep=';',encoding = 'ISO-8859-1')
         










    
     
