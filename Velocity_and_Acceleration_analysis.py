# -*- coding: utf-8 -*-
# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023

# instala a biblioteca função de transferência

# Carrega bibliotecas usadas
import matplotlib.pyplot as plt # gráficos
import numpy as np # funções matemáticas
from scipy import stats # funções estatísticas
import pandas as pd # leitura e manipulação de datasets em vários formatos
from scipy import interpolate
from scipy import signal
import control as ctl
import random
from Openpose_lib_functions import scale_and_offset, align_signals, select_signals_area, smooth_savgol


# Define path e nome dos arquivos a serem lidos
# O path pode mudar de uma conta do drive para outra.
# path = '/content/drive/MyDrive/projetos/lam/data/lemoh/Comparação dos dados (interpolação)/'
path = 'C:/Users/pedro/OneDrive/Documentos/UFPA - Material/TCC STUFF/DATA_TXT_CSV/'
nome_do_arquivo1 = 'Lucas_AbducaoLat_Sentado.csv' # arquivo que contém dados obtidos pelo openpose
nome_do_arquivo2 = 'TESTE 2 - Lucas abdução sentado.txt' #'abduction_Pedro_Trial_grayscale.csv' # arquivo que contém dados obtidos no lemoh

# Lê arquivos de dados do openpose
op = pd.read_csv(path+nome_do_arquivo1) 
# superseded: io.BytesIO(uploaded1[nome_do_arquivo1]))
#op.head(5) # Exibe os primeiros registros da base de dados

# Lê arquivos de dados do lemoh
#lb = pd.read_csv(path + nome_do_arquivo2)
lb = pd.read_table(path+nome_do_arquivo2, decimal = '.', encoding='latin-1') ##needs use enconder 'cause the headers are in pt-br
#superseded: (io.BytesIO(uploaded2[nome_do_arquivo2])), index_col = 0, decimal = ',')


lb_axis_data = lb['Olécrano esq. v(X)'] ## Z for Y axis in LEMOH data
op_axis_data = op['left_elbow_x'] ### gerar csv sem filtro p comparação.
axis_x = 'False' ## If uses the Y axis, the data must be inverted due to LEMOH's equipement calibration
axis_analyzed = str('X')
point_analyzed = str('Olécrano esq. a(X)')

# utilizar sinal com a aplicação do offset
op_data_offset = scale_and_offset(op_axis_data, 'n')

lb_data_offset = scale_and_offset(lb_axis_data, 'n')

if axis_x == 'True':
  lb_data_offset = lb_data_offset*(-1)
else: lb_data_offset = lb_data_offset


N = 2**14 # número de pontos de frequência
fs_a_open = 30 # frenquência de amostragem do openpose (vídeo) ## 30 was the deafult, new has 60fps
# Pode alternar entre os filtros para analisar os resultados.

b,a = signal.iirdesign(wp = 2, ws = 2.5, gpass = 1, 
                         gstop = 30, analog=False, ftype='cheby1', 
                         output='ba', fs= fs_a_open)

w, H = signal.freqz(b = b, a = a, fs = fs_a_open) # Cálculo da resposta em frequência

H_s = ctl.tf(b, a)


op_filtered = signal.lfilter( b = b, a = a, x = op_data_offset ) # operação de filtragem
f, op_data_fft_fil = signal.freqz(op_filtered,worN=N, fs=fs_a_open)


# Define vetores com informação temporal

# Define o vetor tempo de referência (a partir dos dados do lemoh)
time_vec_lb = np.array(lb['Time'])

# Define o vetor de tempo dos dados vindos do openpose
time_vec_op = np.linspace(time_vec_lb.min(), time_vec_lb.max(), len(op))


# Corrige escala e offset do sinal openpose
op_data_offset2 = scale_and_offset(op_filtered, 'n')
f = interpolate.interp1d(time_vec_op,  op_data_offset2, 
                         kind= 'next') # calcula função de interpolação
op_data_interp = f(time_vec_lb) # sinal openpose final (no sense in interpolate again!!!!)



########### Velocity Calculation ###############
op_vel = np.gradient(op_data_interp, time_vec_lb)

#vector:np.array, window_size, poly_order, model
op_vel = smooth_savgol(op_vel, 30,7,'interp') ## 60 for the window_size parameter also works ok

plt.figure()
plt.plot(time_vec_lb, lb_data_offset, label="LEMOH")
plt.plot(time_vec_lb, op_vel, label="Openpose")
plt.title("LEMOH x Openpose Data of " + axis_analyzed + " axis")
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend() 
plt.show()


######### Acceleration Calculation ##############
op_accel = np.gradient(op_vel, time_vec_lb)
op_accel = smooth_savgol(op_accel, 800, 7, 'interp') ## best result was not smoothing out

plt.figure()
plt.plot(time_vec_lb, lb[point_analyzed], label="LEMOH")
plt.plot(time_vec_lb, op_accel, label="Openpose")
plt.title("LEMOH x Openpose Data of " + axis_analyzed + " axis")
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend() 
plt.show()


