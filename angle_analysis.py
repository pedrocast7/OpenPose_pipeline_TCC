import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import stats
from mpl_point_clicker import clicker
from Openpose_lib_functions import scale_and_offset, lowpassfilter, select_signals_area


def scale_and_offset(input_signal: np.array) -> np.array:

  # falta documentar a função
  output_sig = input_signal - np.mean(input_signal) # retira offset
  output =  output_sig/np.max(np.abs(output_sig)) # normaliza amplitude
  return output


columns1 = ["joint_angles_lemoh"]
df = pd.read_csv('/home/lamic/Openpose-pedro/scripts_openpose/codigos/video-01_angles_lemoh.csv', usecols=columns1)

columns2 = ["joint_angles"]
df2 = pd.read_csv('/home/lamic/Openpose-pedro/scripts_openpose/codigos/test_angle.csv', usecols=columns2)

print("Contents in csv file:\n", df)


df = df[1:]
df2 = df2[1:]

lemoh = df.to_numpy()
openpose = df2.to_numpy()

x_lemoh = np.arange( len(lemoh) )

x_openpose_raw = np.arange( len(openpose) )


lhminf, lhmsup, opinf, opsup = select_signals_area(lemoh, openpose)


index_inf_lb = int(np.round(lhminf[0])) # índice inferior do vetor lemoh 
index_sup_lb = int(np.round(lhmsup[0])) # índice super do vetor lemoh
index_inf_op = int(np.round(opinf[0])) # índice inferior do vetor openpose 
index_sup_op = int(np.round(opsup[0])) # índice super do vetor openpose


## Frequency domain analysis


# freq_a = 120 # frequência de amostragem das câmeras do LEMoH (em Hertz)
# N = 2**14 # número de pontos de frequência
# f, op_right_hip_x_fft = signal.freqz(openpose,worN=N, fs=freq_a)
# f, lb_right_hip_x_fft = signal.freqz(lemoh,worN=N, fs=freq_a)


# plt.figure()
# # plt.figure(figsize=(1*6.4,1*4.8)) # inicia nova figura e ajusta tamanho
# plt.plot(f,10*np.log10(np.abs(lb_right_hip_x_fft)), 'r', 
#          label = 'LEMoH') # traça gráfico
# plt.plot(f, 10*np.log10(np.abs(op_right_hip_x_fft)), 'b', 
#          label ='OpenPose') # traça gráfico
# plt.grid('True') # ativa grid
# plt.xlabel('Frequency [Hz]') # legenda do eixo horizontal
# plt.ylabel('Magnitude of the Fourier transform [dB]') # legenda do eixo vertical
# plt.title('Hip - component x') # título do gráfico
# plt.axis([0, 60, -10, 40])
# plt.legend() # exibe legenda
# plt.show()


x_lemoh = np.arange( len(lemoh) )

# Corta os vetores de sinal e tempo
lemoh_trimmed = np.array(lemoh[index_inf_lb:index_sup_lb+1])
x_lemoh_trimmed = np.linspace(x_lemoh[index_inf_lb], x_lemoh[index_sup_lb],
                                  index_sup_lb-index_inf_lb+1) 

openpose_trimmed = np.array(openpose[index_inf_op:index_sup_op+1])
x_openpose_trimmed = np.linspace(x_lemoh_trimmed.min(), x_lemoh_trimmed.max(),
                                  len(openpose_trimmed))



x_openpose = np.arange( len(openpose) )
x_openpose = np.interp(x_openpose, (min(x_openpose), max(x_openpose)), (min(x_lemoh), max(x_lemoh)) )

openpose_trimmed = openpose_trimmed - (np.mean(openpose_trimmed) - np.mean(lemoh_trimmed))
x_openpose_trimmed = x_openpose_trimmed - (np.mean(x_openpose_trimmed) - np.mean(x_lemoh_trimmed))  #offset adjust

#plt.plot(x_openpose_trimmed[index_inf_op], openpose_trimmed[index_inf_op], 'bo')
#plt.plot(x_openpose_trimmed[index_sup_op], openpose_trimmed[index_sup_op], 'bo')

plt.figure()
plt.plot(x_lemoh_trimmed, lemoh_trimmed, label="LEMOH")
plt.plot(x_openpose_trimmed, openpose_trimmed, label="Openpose")
plt.title("LEMOH x Openpose joint angle")
plt.xlabel("Samples")
plt.ylabel("Estimated angle")
plt.show()

openpose_trimmed = np.resize(openpose_trimmed,lemoh_trimmed.size)

f = interpolate.interp1d(np.arange(0, len(openpose_trimmed)), openpose_trimmed)
stretched_openp = f(np.linspace(0.0, len(openpose_trimmed)-1, len(lemoh_trimmed)))





# Calcula sinal de erro
err = np.subtract(lemoh_trimmed,stretched_openp)
err = np.resize(err, (1, len(lemoh_trimmed)))
err = np.transpose(err)

# Calcula SNR signal-to-noise ratio
mse = np.mean(err ** 2) # mean square error
signal_e = np.mean(lemoh_trimmed ** 2) # signal energy
SNR = 10 * np.log10(signal_e/mse)

print('The signal-noise ratio is: ', SNR, ' dB')


lemoh_trimmed_graph = scale_and_offset(lemoh_trimmed)
stretched_openp_graph = scale_and_offset(stretched_openp)
err_graph = scale_and_offset(err)

plt.figure()
plt.plot(x_lemoh_trimmed, lemoh_trimmed_graph,'g', label="LEMOH")
plt.plot(x_openpose_trimmed, stretched_openp_graph[0:len(x_openpose_trimmed)],'b', label="Openpose")
plt.plot(x_lemoh_trimmed, err_graph,'r',label='Error')
plt.title("Error analysis")
plt.xlabel("Samples")
plt.ylabel("Estimated angle")
plt.grid('True')
plt.legend()
plt.show()




#filtered_openpose_15 = lowpassfilter(stretched_openp[0:len(x_openpose_trimmed)], thresh=0.15)
#filtered_openpose_20 = lowpassfilter(stretched_openp[0:len(x_openpose_trimmed)], thresh=0.20)
filtered_openpose_25 = lowpassfilter(stretched_openp[0:len(x_openpose_trimmed)], thresh=0.25)
#filtered_openpose_30 = lowpassfilter(stretched_openp[0:len(x_openpose_trimmed)], thresh=0.30)
#filtered_openpose_35 = lowpassfilter(stretched_openp[0:len(x_openpose_trimmed)], thresh=0.35)


# Savgol Filtering
#filtered_openpose = signal.savgol_filter(stretched_openp[0:len(x_openpose_trimmed)], 51,2)

## Threshold value selection 0.25
# plt.figure()
# plt.subplot(3,2,1)
# plt.plot(x_lemoh_trimmed, lemoh_trimmed, 'r')
# plt.title('Lemoh')
# plt.subplot(3,2,2)
# plt.plot(x_openpose_trimmed, filtered_openpose_15[0:-1], 'b')
# plt.title('Thr = 0.15')
# plt.subplot(3,2,3)
# plt.plot(x_openpose_trimmed, filtered_openpose_20[0:-1], 'b')
# plt.title('Thr = 0.20')
# plt.subplot(3,2,4)
# plt.plot(x_openpose_trimmed, filtered_openpose_25[0:-1], 'b')
# plt.title('Thr = 0.25')
# plt.subplot(3,2,5)
# plt.plot(x_openpose_trimmed, filtered_openpose_30[0:-1], 'b')
# plt.title('Thr = 0.30')
# plt.subplot(3,2,6)
# plt.plot(x_openpose_trimmed, filtered_openpose_35[0:-1], 'b')
# plt.title('Thr = 0.35')
# plt.show()


# Calcula sinal de erro
err2 = np.subtract(lemoh_trimmed,filtered_openpose_25)
err2 = np.resize(err2, (1, len(lemoh_trimmed)))
err = np.transpose(err2)

# Calcula SNR signal-to-noise ratio
mse2 = np.mean(err2 ** 2) # mean square error
signal_e2 = np.mean(lemoh_trimmed ** 2) # signal energy
SNR2 = 10 * np.log10(signal_e2/mse2)

print('The signal-noise ratio, after the filtering process, is: ', SNR2, ' dB')


plt.figure()
plt.plot(x_lemoh_trimmed, lemoh_trimmed,'g', label="LEMOH")
plt.plot(x_openpose_trimmed, filtered_openpose_25[0:len(x_openpose_trimmed)],'b', label="Openpose")
plt.title("Filtered Openpose")
plt.xlabel("Samples")
plt.ylabel("Estimated angle")
plt.grid('True')
plt.legend()
plt.show()


## CORRELATION MEASUREMENT 

arr2_interp = interpolate.interp1d(np.arange(filtered_openpose_25.size),filtered_openpose_25)
openpose2_stretch = arr2_interp(np.linspace(0,filtered_openpose_25.size-1,lemoh_trimmed.size))

# f = interpolate.interp1d(np.arange(0, len(filtered_openpose_25)), filtered_openpose_25)
# stretched_openp = f(np.linspace(0.0, len(filtered_openpose_25)-1, len(lemoh_trimmed)))

##find line of best fit
a_fit, b_fit = np.polyfit(openpose2_stretch, lemoh_trimmed, 1)

plt.figure()

##add line of best fit to plot
plt.plot(openpose2_stretch, a_fit*openpose2_stretch+b_fit, color='red', linestyle='--')

plt.scatter(openpose2_stretch, lemoh_trimmed)
plt.grid('True')
plt.xlabel('OpenPose inference')
plt.ylabel('LEMOH inference')
plt.title('Angle estimation')
plt.savefig('cloud_OpLh_Angle.eps', format='eps')
plt.show()

## calculating cross corelation and Pearson's corrcoef

Pear_coef = stats.pearsonr(np.array(np.squeeze(lemoh_trimmed)), np.array(np.squeeze(openpose2_stretch)))

# #Cross_coef = signal.correlate(op_right_hip_x_final, lb_right_hip_x_final)

plt.figure()
plt.plot(Pear_coef, color='red', linestyle='--')
plt.grid('True')
plt.title('Angle Pearson\'s Correlation = ' + "{:.4f}".format(Pear_coef[0]))
plt.savefig('Pearsons_correlation_OpLh_Angle.eps', format='eps')
plt.show()


plt.figure()
x_corr = plt.xcorr(np.squeeze(lemoh_trimmed), np.squeeze(openpose2_stretch), usevlines=True, maxlags=100, normed=True, lw=2)
# #plt.xcorr(lb_right_hip_x_final, op_right_hip_x_final, usevlines=True, maxlags=58, normed=True, lw=2)
plt.grid('True')
plt.title('Angle Cross-correlation')
plt.xlabel('Lags')
plt.ylabel('Normalized Cross-correlation')
plt.savefig('Cross_correlation_OpLh_Angle.eps', format='eps')
plt.show()

lags = x_corr[0]
c_values = x_corr[1]


print(Pear_coef)
print(a_fit,b_fit)
print('The lag value for the highest Xcorrelation is {}'.format(lags[np.argmax(c_values)]) + ', giving the {:.4f} score'.format(c_values[np.argmax(c_values)]))