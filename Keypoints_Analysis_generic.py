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
from Openpose_lib_functions import scale_and_offset, align_signals, select_signals_area


# Define path e nome dos arquivos a serem lidos
# O path pode mudar de uma conta do drive para outra.
# path = '/content/drive/MyDrive/projetos/lam/data/lemoh/Comparação dos dados (interpolação)/'
path = '/home/lamic/Openpose-pedro/LEMOH_EXP/'
nome_do_arquivo1 = 'cam1_op_video_grayscale_new.csv' # arquivo que contém dados obtidos pelo openpose
nome_do_arquivo2 = 'testeopenpose.txt' #'abduction_Pedro_Trial_grayscale.csv' # arquivo que contém dados obtidos no lemoh

# Lê arquivos de dados do openpose
op = pd.read_csv(path+nome_do_arquivo1) 
# superseded: io.BytesIO(uploaded1[nome_do_arquivo1]))
op.head(5) # Exibe os primeiros registros da base de dados

# Lê arquivos de dados do lemoh
#lb = pd.read_csv(path + nome_do_arquivo2)
lb = pd.read_table(path+nome_do_arquivo2, decimal = ',', encoding='latin-1') ##needs use enconder 'cause the headers are in pt-br
#superseded: (io.BytesIO(uploaded2[nome_do_arquivo2])), index_col = 0, decimal = ',')
lb.head(5) # Exibe os primeiros registros da base de dados



  # lag_timesteps = int(round(lag/timestep))

  #  # compute new values for t, a & b for plotting
  # if lag > 0:
  #       new_a = list(a) + [np.nan]*lag_timesteps
  #       new_b = [np.nan]*lag_timesteps + list(b)
  #       new_t = np.linspace(0, (len(time_vec)-1)+lag, int(round(((len(time_vec)-1)+lag)/timestep)))
  # else:
  #       new_a = [np.nan]*abs(lag_timesteps) + list(a)
  #       new_b = list(b) + [np.nan]*abs(lag_timesteps)
  #       new_t = np.linspace(0+lag, (len(time_vec)-1), int(round(((len(time_vec)-1)-(0+lag))/timestep)))
  # return new_a, new_b, new_t


lb_left_shoulder_x = lb['Olécrano esq. X']
op_left_shoulder_x = op['left_elbow_x'] ### gerar csv sem filtro p comparação.
axis_x = 'True' ## If uses the X axis, the data must be inverted due to LEMOH's equipement calibration


# utilizar sinal com a aplicação do offset
op_left_shoulder_x_offset = scale_and_offset(op_left_shoulder_x, 'n')

lb_left_shoulder_x_offset = scale_and_offset(lb_left_shoulder_x, 'n')

if axis_x == 'True':
  lb_left_shoulder_x_offset = lb_left_shoulder_x_offset*(-1)
else: lb_left_shoulder_x_offset = lb_left_shoulder_x_offset


#op_left_shoulder_x_offset = op_left_shoulder_x       #### RAW SIGNAL

#lb_left_shoulder_x_offset = lb_left_shoulder_x       #### RAW SIGNAL

N = 2**14 # número de pontos de frequência
fs_a_open = 60 # frenquência de amostragem do openpose (vídeo) ## 30 was the deafult, new has 60fps
# Pode alternar entre os filtros para analisar os resultados.

b,a = signal.iirdesign(wp = 2, ws = 2.5, gpass = 1, 
                         gstop = 30, analog=False, ftype='cheby1', 
                         output='ba', fs= fs_a_open)

w, H = signal.freqz(b = b, a = a, fs = fs_a_open) # Cálculo da resposta em frequência

H_s = ctl.tf(b, a)

print('Ordem do filtro: ', len(b)-1)
print('Coeficientes do denominador da função de transferência: ', a)
print('Coeficientes do numerador da função de transferência: ',b)
print('Função de transferência: ', H_s)


op_left_shoulder_x_fil = signal.lfilter( b = b, a = a, x = op_left_shoulder_x_offset ) # operação de filtragem
f, op_left_shoulder_x_fft_fil = signal.freqz(op_left_shoulder_x_fil,worN=N, fs=fs_a_open)

#scale = 0.5
plt.figure()
# plt.figure(figsize=(scale*6.4,scale*4.8))
plt.grid(True)
plt.plot(w, 20 * np.log10(abs(H)))
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Amplitude [dB]')
plt.axis([0, 8, -60, 1])
plt.title('Magnitude frequency response')
plt.savefig('filtro.eps', format='eps')
plt.show()

plt.figure()
plt.plot(range(len(op_left_shoulder_x_offset)), op_left_shoulder_x_offset,'b' ,label="Openpose")
plt.plot(range(len(lb_left_shoulder_x_offset)), lb_left_shoulder_x_offset, 'r', label="LEMOH")
plt.title("LEMOH x Openpose Data of X axis RAW")
plt.xlabel("Frames")
plt.ylabel("Displacement")
plt.grid()
plt.legend() # exibe legenda
plt.show()



# plt.figure()
# # plt.figure(figsize=(scale*6.4,scale*4.8))
# plt.grid(True)
# plt.plot(w, 20 * np.log10(abs(H)))
# plt.xlabel('Frequency [rad/sample]')
# plt.ylabel('Amplitude [dB]')
# plt.axis([0, 8, -60, 1])
# plt.title('Magnitude frequency response')
# #plt.savefig('filtro.eps', format='eps')
# plt.show()

plt.figure()
# plt.figure(figsize=(scale*6.4,scale*4.8))
plt.plot(range(len(op_left_shoulder_x)),op_left_shoulder_x, 'b', 
         label = 'OpenPose') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frames') # legenda do eixo horizontal
plt.ylabel('Component X') # legenda do eixo vertical
plt.title('Left Elbow') # título do gráfico
plt.legend() # exibe legenda
#plt.savefig('entrada_op.eps', format='eps')
plt.show()

plt.figure()
# plt.figure(figsize=(scale*6.4,scale*4.8))
plt.plot(range(len(lb_left_shoulder_x)),lb_left_shoulder_x, 'r', 
         label = 'LEMOH') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frames') # legenda do eixo horizontal
plt.ylabel('Component X') # legenda do eixo vertical
plt.title('Left Elbow') # título do gráfico
plt.legend() # exibe legenda
#plt.savefig('entrada_lb.eps', format='eps')
plt.show()




plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8))
plt.plot(range(len(op_left_shoulder_x_fil)),op_left_shoulder_x_fil, 'g', 
         label = 'OpenPose (filtered)') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frames') # legenda do eixo horizontal
plt.ylabel('Component X') # legenda do eixo vertical
plt.title('Left Elbow') # título do gráfico
plt.legend() # exibe legenda
#plt.savefig('filtrado_op.eps', format='eps')
plt.show()


plt.figure()
# plt.figure(figsize=(0.5*6.4,0.5*4.8))
plt.plot(range(len(lb_left_shoulder_x_offset)),lb_left_shoulder_x_offset, 'r') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frames') # legenda do eixo horizontal
plt.ylabel('Displacement (m)') # legenda do eixo vertical
plt.title('Left Elbow') # título do gráfico
plt.legend() # exibe legenda
#plt.savefig('ajuste_lb.eps', format='eps')
plt.show()

plt.figure()
# plt.figure(figsize=(0.5*6.4,0.5*4.8))
plt.plot(range(len(op_left_shoulder_x_offset)),op_left_shoulder_x_offset, 'b') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frames') # legenda do eixo horizontal
plt.ylabel('Displacement (m)') # legenda do eixo vertical
plt.title('Left Elbow') # título do gráfico
plt.legend() # exibe legenda
#plt.savefig('ajuste_op.eps', format='eps')
plt.show()

plt.figure()
# plt.figure(figsize=(scale*6.4,scale*4.8))
plt.plot(range(len(op_left_shoulder_x_offset)),op_left_shoulder_x_offset, 'b', 
         label = 'OpenPose (initial)')
plt.plot(range(len(op_left_shoulder_x_fil)),op_left_shoulder_x_fil, 'magenta', 
         label = 'OpenPose filtered (initial)')
plt.grid('True') # ativa grid
plt.xlabel('Frames') # legenda do eixo horizontal
plt.ylabel('Displacement (m)') # legenda do eixo vertical
plt.title('Left Elbow') # título do gráfico
plt.legend() # exibe legenda
plt.show()

# Define vetores com informação temporal

# Define o vetor tempo de referência (a partir dos dados do lemoh)
time_vec_lb = np.array(lb['Time'])

# Define o vetor de tempo dos dados vindos do openpose
time_vec_op = np.linspace(time_vec_lb.min(), time_vec_lb.max(), len(op))

lemoh = lb_left_shoulder_x_offset

openpose = op_left_shoulder_x_fil

#time_vec_lb = np.arange( len(lemoh) )

#x_openpose_raw = np.arange( len(openpose) )


lhminf, lhmsup, opinf, opsup = select_signals_area(lemoh, openpose)


index_inf_lb = int(np.round(lhminf[0])) # índice inferior do vetor lemoh 
index_sup_lb = int(np.round(lhmsup[0])) # índice super do vetor lemoh
index_inf_op = int(np.round(opinf[0])) # índice inferior do vetor openpose 
index_sup_op = int(np.round(opsup[0])) # índice super do vetor openpose


# Corta os vetores de sinal e tempo
lemoh_trimmed = np.array(lemoh[index_inf_lb:index_sup_lb+1])
time_vec_lb_trimmed = np.linspace(time_vec_lb[index_inf_lb], time_vec_lb[index_sup_lb],
                                  index_sup_lb-index_inf_lb+1) 

openpose_trimmed = np.array(openpose[index_inf_op:index_sup_op+1])
time_vec_op_trimmed = np.linspace(time_vec_lb_trimmed.min(), time_vec_lb_trimmed.max(),
                                  len(openpose_trimmed))



# x_openpose = np.arange( len(openpose) )
# x_openpose = np.interp(x_openpose, (min(x_openpose), max(x_openpose)), (min(time_vec_lb), max(time_vec_lb)) )

#openpose_trimmed = openpose_trimmed - (np.mean(openpose_trimmed) - np.mean(lemoh_trimmed))
#time_vec_op_trimmed = time_vec_op_trimmed - (np.mean(time_vec_op_trimmed) - np.mean(time_vec_lb_trimmed))  #offset adjust

plt.figure()
plt.plot(time_vec_lb_trimmed, lemoh_trimmed, label="LEMOH")
plt.plot(time_vec_op_trimmed, openpose_trimmed, label="Openpose")
plt.title("LEMOH x Openpose Data of X axis")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend() # exibe legenda
plt.show()

# Corrige escala e offset do sinal openpose
op_x_corrigido = scale_and_offset(openpose_trimmed, 'n')
f = interpolate.interp1d(time_vec_op_trimmed,  op_x_corrigido, 
                         kind= 'next') # calcula função de interpolação
op_x_final = f(time_vec_lb_trimmed) # sinal openpose final (no sense in interpolate again!!!!)

# Corrige escala e offset do sinal lemoh
lb_x_final = scale_and_offset(lemoh_trimmed, 'n') # sinal lemoh final



# err = np.array(err)
# 
# print(err.shape)
# 
# plt.figure()
# plt.plot(time_vec_lb_trimmed, err)
# plt.show()



#plt.savefig('comparacao_tempo.eps', format='eps')


## SIGNAL ALIGNMENT

aligned_op = align_signals(lb_x_final, op_x_final)

plt.figure()
plt.plot(time_vec_lb_trimmed, lb_x_final, label="LEMOH")
plt.plot(time_vec_lb_trimmed, aligned_op, label="Openpose")
plt.title("Openpose Data aligned")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.grid('True')
plt.legend() # exibe legenda
plt.show()


# Calcula sinal de erro
err = lb_x_final - op_x_final

# SNR signal-to-noise ratio
mse = np.mean(err ** 2) # mean square error
signal_e = np.mean(lb_x_final ** 2) # signal energy
SNR = 10 * np.log10(signal_e/mse)

print('A razão sinal-ruído é de: ', SNR, ' dB')

# Traça gráficos
#scale = 1.5
plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(time_vec_lb_trimmed,aligned_op, 'b', 
         label ='OpenPose (final)') # traça gráfico
plt.plot(time_vec_lb_trimmed,lb_x_final, 'r', 
         label = 'LEMoH (final)') # traça gráfico
plt.plot(time_vec_lb_trimmed,err, 'y', 
         label = 'Erro (final)') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Time [s]') # legenda do eixo horizontal
plt.ylabel('Component X') # legenda do eixo vertical
plt.title('X axis error analysis') # título do gráfico
plt.legend() # exibe legenda
plt.show()


#### REGRESSION OF INFERENCE
# random_sample_list = random.choices(time_vec_lb_trimmed, k=10) #takes 40 random points
# lb_points = lb_x_final[random_sample_list]
# op_points = aligned_op[random_sample_list]
lb_points, op_points = zip(*random.sample(list(zip(lb_x_final, aligned_op)), 40))

a_val_fit, b_val_fit = np.polyfit(op_x_final, lb_x_final, 1)

plt.figure()
plt.plot(aligned_op, a_val_fit*aligned_op+b_val_fit, color='red')
plt.scatter(op_points, lb_points)
plt.title('Random points inference validation')
plt.xlabel("Openpose inference")
plt.ylabel("LEMOH inference")
plt.grid('True')
plt.show()

## CORRELATION MEASUREMENT 

#find line of best fit
a_fit, b_fit = np.polyfit(op_x_final, lb_x_final, 1)

plt.figure()

#add line of best fit to plot
plt.plot(op_x_final, a_fit*op_x_final+b_fit, color='red', linestyle='--')

plt.scatter(op_x_final, lb_x_final)
plt.grid('True')
plt.xlabel('OpenPose inference')
plt.ylabel('LEMOH inference')
plt.title('Scatter of Y component (m)')
#plt.savefig('cloud_OpLh.eps', format='eps')
plt.show()

## calculating cross corelation and Pearson's corrcoef

Pear_coef = stats.pearsonr(lb_x_final, op_x_final)

#Cross_coef = signal.correlate(op_x_final, lb_right_hip_x_final)

plt.figure()
plt.plot(Pear_coef, color='red', linestyle='--')
plt.grid('True')
plt.title('Y Pearson\'s Correlation = ' + "{:.4f}".format(Pear_coef[0]))
#plt.savefig('Pearsons_correlation_OpLh.eps', format='eps')
plt.show()

plt.figure()

x_corr = plt.xcorr(lb_x_final, op_x_final, usevlines=True, maxlags=200, normed=True, lw=2)
#plt.xcorr(lb_right_hip_x_final, op_x_final, usevlines=True, maxlags=58, normed=True, lw=2)
plt.grid('True')
plt.title('Y axis data Cross-correlation')
plt.xlabel('Lags')
plt.ylabel('Normalized Cross-correlation')
#plt.savefig('Cross_correlation_OpLh.eps', format='eps')
plt.show()

lags = x_corr[0]
c_values = x_corr[1]


print(Pear_coef)
print(a_fit,b_fit)
print('The lag value for the highest Xcorrelation is {}'.format(lags[np.argmax(c_values)]) + ', giving the {:.4f} score'.format(c_values[np.argmax(c_values)]))

# Avaliação no domínio da frequência

freq_a = 120 # frequência de amostragem das câmeras do LEMoH (em Hertz)
N = 2**14 # número de pontos de frequência
f, op_x_data_fft = signal.freqz(op_x_final,worN=N, fs=freq_a)
f, lb_x_data_fft = signal.freqz(lb_x_final,worN=N, fs=freq_a)
f, err_fft = signal.freqz(err,worN=N, fs=freq_a)

# Traça gráficos

#scale = 1.5
plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(f,10*np.log10(np.abs(lb_x_data_fft)), 'r', 
         label = 'LEMoH') # traça gráfico
plt.plot(f, 10*np.log10(np.abs(op_x_data_fft)), 'b', 
         label ='OpenPose') # traça gráfico
#plt.semilogx(f,10*np.log(np.abs(err_fft)), 'y', label = 'Erro') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
plt.title('Component Y: Lemoh & OpenPose') # título do gráfico
plt.axis([0, 60, -10, 40])
plt.legend() # exibe legenda
#plt.savefig('comparacao_freq.eps', format='eps')
plt.show()

plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(f,10*np.log10(np.abs(lb_x_data_fft)), 'r', 
         label = 'LEMoH') # traça gráfico
plt.plot(f, 10*np.log10(np.abs(op_x_data_fft)), 'b', 
         label ='OpenPose') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frequência [Hz]') # legenda do eixo horizontal
plt.ylabel('Magnitude da transformada de Fourier [dB]') # legenda do eixo vertical
plt.title('Component X: Lemoh & OpenPose') # título do gráfico
plt.axis([0, 1.3, -10, 40])
plt.legend() # exibe legenda

# plt.figure()
# # plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
# plt.plot(f,10*np.log10(np.abs(lb_x_data_fft)), 'r', 
#          label = 'LEMoH') # traça gráfico
# plt.plot(f, 10*np.log10(np.abs(op_x_data_fft)), 'b', 
#          label ='OpenPose') # traça gráfico
# plt.grid('True') # ativa grid
# plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
# plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
# plt.title('Component X: Lemoh & OpenPose') # título do gráfico
# plt.axis([3, 4, -10, 40])
# plt.legend() # exibe legenda

# plt.figure()
# # plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
# plt.plot(f,10*np.log10(np.abs(lb_x_data_fft)), 'r', 
#          label = 'LEMoH') # traça gráfico
# plt.plot(f, 10*np.log10(np.abs(op_x_data_fft)), 'b', 
#          label ='OpenPose') # traça gráfico
# plt.grid('True') # ativa grid
# plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
# plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
# plt.title('Component X: Lemoh & OpenPose') # título do gráfico
# plt.axis([4, 5, -10, 40])
# plt.legend() # exibe legenda

# plt.figure()
# # plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
# plt.plot(f,10*np.log10(np.abs(lb_x_data_fft)), 'r', 
#          label = 'LEMoH') # traça gráfico
# plt.plot(f, 10*np.log10(np.abs(op_x_data_fft)), 'b', 
#          label ='OpenPose') # traça gráfico
# plt.grid('True') # ativa grid
# plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
# plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
# plt.title('Component X: Lemoh & OpenPose') # título do gráfico
# plt.axis([0, 60, -10, 40])
# plt.legend() # exibe legenda

# plt.show()

# SNR no domínio da frequência
plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(f,20*np.log10(np.abs(lb_x_data_fft)) - 
                     20*np.log10(np.abs(err_fft)), 'k', label = 'SNR') # traça gráfico
#plt.plot(f, 10*np.log(np.abs(op_x_data_fft)), 'b', label ='OpenPose') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
plt.ylabel('Signal-to-noise ratio [dB]') # legenda do eixo vertical
plt.title('Component Y: Lemoh & OpenPose') # título do gráfico
plt.axis([0, 60 , -50, 50])
plt.legend() # exibe legenda
#plt.savefig('SNR.eps', format='eps')
plt.show()

plt.figure()
# plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
plt.plot(f,10*np.log10(np.abs(lb_x_data_fft)), 'r', 
         label = 'LEMoH') # traça gráfico
plt.plot(f, 10*np.log10(np.abs(err_fft)), 'g', 
         label ='Erro') # traça gráfico
plt.grid('True') # ativa grid
plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
plt.title('Component Y: Lemoh & OpenPose') # título do gráfico
plt.axis([0, 5, -10, 40])
plt.legend() # exibe legenda
#plt.savefig('erro.eps', format='eps')

plt.show()

