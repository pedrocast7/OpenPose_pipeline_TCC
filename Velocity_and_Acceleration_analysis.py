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
axis_flip = 'False' ## If uses the Y axis, the data must be inverted due to LEMOH's equipement calibration
axis_analyzed = str('X')
point_analyzed = str('Olécrano esq. a(X)')
lemoh_fps = 120

# utilizar sinal com a aplicação do offset
op_data_offset = scale_and_offset(op_axis_data, 'n')

lb_data_offset = scale_and_offset(lb_axis_data, 'n')

if axis_flip == 'True':
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
print(f"displa: {len(op_data_interp)} and time len: {len(time_vec_lb)}")
dv = np.gradient(op_data_interp)
dt = 1/lemoh_fps
op_vel_raw = dv/dt
#op_vel_raw = np.gradient(op_data_interp, time_vec_lb)

print(f"vel len: {len(op_vel_raw)} and time len: {len(time_vec_lb)}")

#vector:np.array, window_size, poly_order, model
op_vel = smooth_savgol(op_vel_raw, 30,7,'interp') ## 60 for the window_size parameter also works ok

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
da = np.gradient(op_vel_raw)
op_accel_raw = da/dt

#op_accel_raw = np.gradient(op_vel, time_vec_lb)
print(f"accel len: {len(op_accel_raw)} and time len: {len(time_vec_lb)}")
op_accel = smooth_savgol(op_accel_raw, 35, 7, 'interp') ## 800, 7 best result was not smoothing out

plt.figure()
plt.plot(time_vec_lb, lb[point_analyzed], label="LEMOH")
plt.plot(time_vec_lb, op_accel, label="Openpose")
plt.title("LEMOH x Openpose Data of " + axis_analyzed + " axis")
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend() 
plt.show()


####################################################################################################


############### Velocity analysis ##########

lhminf, lhmsup, opinf, opsup = select_signals_area(lb_data_offset, op_vel)


index_inf_lb = int(np.round(lhminf[0])) # índice inferior do vetor lemoh 
index_sup_lb = int(np.round(lhmsup[0])) # índice super do vetor lemoh
index_inf_op = int(np.round(opinf[0])) # índice inferior do vetor openpose 
index_sup_op = int(np.round(opsup[0])) # índice super do vetor openpose


# Corta os vetores de sinal e tempo
lemoh_vel_trimmed = np.array(lb_data_offset[index_inf_lb:index_sup_lb+1])
time_vec_lb_trimmed = np.linspace(time_vec_lb[index_inf_lb], time_vec_lb[index_sup_lb],
                                  index_sup_lb-index_inf_lb+1) 

openpose_vel_trimmed = np.array(op_vel[index_inf_op:index_sup_op+1])
time_vec_op_trimmed = np.linspace(time_vec_lb_trimmed.min(), time_vec_lb_trimmed.max(),
                                  len(openpose_vel_trimmed))


## Trimmed signals

plt.figure()
plt.plot(time_vec_lb_trimmed, lemoh_vel_trimmed, label="LEMOH")
plt.plot(time_vec_op_trimmed, openpose_vel_trimmed, label="Openpose")
plt.title("LEMOH x Openpose Velocity of " + axis_analyzed + " axis (No Alignment)")
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend() 
plt.show()


# Corrige escala e offset do sinal openpose
op_vel_offset = scale_and_offset(openpose_vel_trimmed, 'n')
f = interpolate.interp1d(time_vec_op_trimmed,  op_vel_offset, 
                         kind= 'next') # calcula função de interpolação
op_vel_final = f(time_vec_lb_trimmed) # sinal openpose final (no sense in interpolate again!!!!)

# Corrige escala e offset do sinal lemoh
lb_vel_final = scale_and_offset(lemoh_vel_trimmed, 'n') # sinal lemoh final



#### Velocity Alignment ####

aligned_op_vel = align_signals(lb_vel_final, op_vel_final)

plt.figure()
plt.plot(time_vec_lb_trimmed, lb_vel_final, label="LEMOH")
plt.plot(time_vec_lb_trimmed, aligned_op_vel, label="Openpose")
plt.title("Openpose Data Alignment")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid('True')
plt.legend() # exibe legenda
plt.show()


### Velocity Error #####
# Calcula sinal de erro
err = lb_vel_final - op_vel_final

# SNR signal-to-noise ratio
mse = np.mean(err ** 2) # mean square error
signal_e = np.mean(lb_vel_final ** 2) # signal energy
SignalNR = 10 * np.log10(signal_e/mse)
print('A razão sinal-ruído é de: ', SignalNR, ' dB')


plt.figure()
plt.plot(time_vec_lb_trimmed,aligned_op_vel, 'b', 
         label ='OpenPose (final)')
plt.plot(time_vec_lb_trimmed,lb_vel_final, 'r', 
         label = 'LEMoH (final)')
plt.plot(time_vec_lb_trimmed,err, 'y', 
         label = 'Error (final)')
plt.grid('True') 
plt.xlabel('Time (s)') 
plt.ylabel('Velocity (m/s)') 
plt.title(axis_analyzed + ' axis Error Analysis') # título do gráfico
plt.legend() # exibe legenda
plt.show()



#### Velocity Regression of Inference
# random_sample_list = random.choices(time_vec_lb_trimmed, k=10) #takes 40 random points
# lb_points = lb_x_final[random_sample_list]
# op_points = aligned_op[random_sample_list]
lb_points, op_points = zip(*random.sample(list(zip(lb_vel_final, aligned_op_vel)), 40))

a_val_fit, b_val_fit = np.polyfit(op_vel_final, lb_vel_final, 1)

plt.figure()
plt.plot(aligned_op_vel, a_val_fit*aligned_op_vel+b_val_fit, color='red')
plt.scatter(op_points, lb_points)
plt.title('Random points inference validation')
plt.xlabel("Openpose inference")
plt.ylabel("LEMOH inference")
plt.grid('True')
plt.show()

## Full Scatter plot
#find line of best fit
a_fit, b_fit = np.polyfit(op_vel_final, lb_vel_final, 1)

plt.figure()

#add line of best fit to plot
plt.plot(op_vel_final, a_fit*op_vel_final+b_fit, color='red', linestyle='--')

plt.scatter(aligned_op_vel, lb_vel_final)
plt.grid('True')
plt.xlabel('OpenPose inference')
plt.ylabel('LEMOH inference')
plt.title('Scatter of ' + axis_analyzed + ' Velocity component (m/s)')
#plt.savefig('cloud_OpLh.eps', format='eps')
plt.show()


#### Correlation Measurement

## calculating cross corelation and Pearson's corrcoef

Pear_coef = stats.pearsonr(lb_vel_final, op_vel_final)
print(f" The Pearson's Correlation Value is: {Pear_coef}")


x_corr = plt.xcorr(lb_vel_final, op_vel_final, usevlines=True, maxlags=200, normed=True, lw=2)
#plt.xcorr(lb_right_hip_x_final, op_x_final, usevlines=True, maxlags=58, normed=True, lw=2)
plt.grid('True')
plt.title(axis_analyzed + ' axis data Cross-correlation')
plt.xlabel('Lags')
plt.ylabel('Normalized Cross-correlation')
#plt.savefig('Cross_correlation_OpLh.eps', format='eps')
plt.show()

lags = x_corr[0]
c_values = x_corr[1]

#print(a_fit,b_fit)
print('The lag value for the highest Xcorrelation is {}'.format(lags[np.argmax(c_values)]) + ', giving the {:.4f} score'.format(c_values[np.argmax(c_values)]))




### Velocity Frequency Analysis ###

freq_a = 120 # frequência de amostragem das câmeras do LEMoH (em Hertz)
N = 2**14 # número de pontos de frequência
f, op_vel_data_fft = signal.freqz(op_vel_final,worN=N, fs=freq_a)
f, lb_vel_data_fft = signal.freqz(lb_vel_final,worN=N, fs=freq_a)
f, err_fft = signal.freqz(err,worN=N, fs=freq_a)

# Traça gráficos

#scale = 1.5
plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(f,10*np.log10(np.abs(lb_vel_data_fft)), 'r', 
         label = 'LEMOH') # traça gráfico
plt.plot(f, 10*np.log10(np.abs(op_vel_data_fft)), 'b', 
         label ='OpenPose') # traça gráfico
#plt.semilogx(f,10*np.log(np.abs(err_fft)), 'y', label = 'Erro') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
plt.title(f"Component {axis_analyzed}: Lemoh & OpenPose") # título do gráfico
plt.axis([0, 60, -10, 40])
plt.legend() # exibe legenda
#plt.savefig('comparacao_freq.eps', format='eps')
plt.show()

plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(f,10*np.log10(np.abs(lb_vel_data_fft)), 'r', 
         label = 'LEMoH') # traça gráfico
plt.plot(f, 10*np.log10(np.abs(op_vel_data_fft)), 'b', 
         label ='OpenPose') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frequência [Hz]') # legenda do eixo horizontal
plt.ylabel('Magnitude da transformada de Fourier [dB]') # legenda do eixo vertical
plt.title(f"Component {axis_analyzed}: Lemoh & OpenPose") # título do gráfico
plt.axis([0, 1.3, -10, 40])
plt.legend() # exibe legenda
plt.show()



########### Acceleration Analysis ###########

