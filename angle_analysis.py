# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import stats
from mpl_point_clicker import clicker
from Openpose_lib_functions import scale_and_offset, lowpassfilter, select_signals_area, align_signals
from sklearn.metrics import mean_squared_error



columns1 = ["joint_angles_lemoh"]
df = pd.read_csv('C:/Users/pedro/OneDrive/Documentos/UFPA - Material/TCC STUFF/lamic/Samples_04-10/DATA_TXT_CSV/LB_angle_jessyka_abducao_lat.csv')

columns2 = ["joint_angles"]
df2 = pd.read_csv('C:/Users/pedro/OneDrive/Documentos/UFPA - Material/TCC STUFF/lamic/Samples_04-10/DATA_TXT_CSV/OP_angle_jessyka_abducao_lat.csv')

print("Contents in csv file:\n", df)

lb = pd.read_table('C:/Users/pedro/OneDrive/Documentos/UFPA - Material/TCC STUFF/lamic/Samples_04-10/DATA_TXT_CSV/jessica abdução lateral KINEM.txt',
                    decimal = '.',
                      encoding='latin-1')

df = df[1:]
df2 = df2[1:]

lemoh = df.to_numpy()
openpose = df2.to_numpy() # shape(n_frames, 1)

## Turn to 1D vec
lemoh = lemoh.reshape(-1)
openpose = openpose.reshape(-1)

# Define vetores com informação temporal

# Define o vetor tempo de referência (a partir dos dados do lemoh)
time_vec_lb = np.array(lb['Time'])

# Define o vetor de tempo dos dados vindos do openpose
time_vec_op = np.linspace(time_vec_lb.min(), time_vec_lb.max(), len(openpose))

f = interpolate.interp1d(time_vec_op,  openpose, 
                         kind= 'next') # calcula função de interpolação
openpose = f(time_vec_lb)

#openpose = scale_and_offset(openpose, 'n')
#lemoh = scale_and_offset(lemoh, 'n')

#time_vec_lb = np.arange( len(lemoh) )

#time_vec_op = np.arange( len(openpose) )


lhminf, lhmsup, opinf, opsup = select_signals_area(lemoh, openpose)


index_inf_lb = int(np.round(lhminf[0])) # índice inferior do vetor lemoh 
index_sup_lb = int(np.round(lhmsup[0])) # índice super do vetor lemoh
index_inf_op = int(np.round(opinf[0])) # índice inferior do vetor openpose 
index_sup_op = int(np.round(opsup[0])) # índice super do vetor openpose




# Corta os vetores de sinal e tempo
lemoh_trimmed = np.array(lemoh[index_inf_lb:index_sup_lb+1])
x_lemoh_trimmed = np.linspace(time_vec_lb[index_inf_lb], time_vec_lb[index_sup_lb],
                                  index_sup_lb-index_inf_lb+1) 

openpose_trimmed = np.array(openpose[index_inf_op:index_sup_op+1])
x_openpose_trimmed = np.linspace(x_lemoh_trimmed.min(), x_lemoh_trimmed.max(),
                                  len(openpose_trimmed))





#x_openpose = np.arange( len(openpose) )
#x_openpose = np.interp(x_openpose, (min(x_openpose), max(x_openpose)), (min(time_vec_lb), max(time_vec_lb)) )

#openpose_trimmed = openpose_trimmed - (np.mean(openpose_trimmed) - np.mean(lemoh_trimmed))
#x_openpose_trimmed = x_openpose_trimmed - (np.mean(x_openpose_trimmed) - np.mean(x_lemoh_trimmed))  #offset adjust


#plt.plot(x_openpose_trimmed[index_inf_op], openpose_trimmed[index_inf_op], 'bo')
#plt.plot(x_openpose_trimmed[index_sup_op], openpose_trimmed[index_sup_op], 'bo')

plt.figure()
plt.plot(x_lemoh_trimmed, lemoh_trimmed, label="LEMOH")
plt.plot(x_openpose_trimmed, openpose_trimmed, label="Openpose")
plt.title("LEMOH x Openpose joint angle")
plt.xlabel("Time (s)")
plt.ylabel("Estimated angle")
plt.grid()
plt.show()

#openpose_trimmed = np.resize(openpose_trimmed,lemoh_trimmed.size)


### Fill NaN values that may be lost after trim
openpose_trimmed = openpose_trimmed[~np.isnan(openpose_trimmed)]

f = interpolate.interp1d(np.arange(0, len(openpose_trimmed)), openpose_trimmed)
openpose_trimmed = f(np.linspace(0.0, len(openpose_trimmed)-1, len(lemoh_trimmed)))

#f = interpolate.interp1d(openpose_trimmed, 
#                         kind= 'next') # calcula função de interpolação
#openpose_trimmed = f(x_lemoh_trimmed)



# Calcula sinal de erro
err = lemoh_trimmed - openpose_trimmed
#err = np.resize(err, (1, len(lemoh_trimmed)))
err = err.reshape(-1) #1d

# Calcula SNR signal-to-noise ratio
#mse = mean_squared_error(lemoh_trimmed, openpose_trimmed) # mean square error
mse = np.mean(err ** 2)
signal_e = np.mean(lemoh_trimmed ** 2) # signal energy
SignalNR = 10 * np.log10(signal_e/mse)

print('The signal-noise ratio is: ', SignalNR, ' dB')


#lemoh_trimmed = scale_and_offset(lemoh_trimmed, 'n')
#openpose_trimmed = openpose_trimmed - (np.mean(openpose_trimmed) - np.mean(lemoh_trimmed))
#openpose_trimmed = scale_and_offset(openpose_trimmed, 'n')

err_plot = err + np.mean(lemoh_trimmed) ## offset adjust for plotting

plt.figure()
plt.plot(x_lemoh_trimmed, lemoh_trimmed,'g', label="LEMOH")
plt.plot(x_lemoh_trimmed, openpose_trimmed,'b', label="Openpose")
plt.plot(x_lemoh_trimmed, err_plot, 'r',label='Error')
plt.title("Error analysis")
plt.xlabel("Samples")
plt.ylabel("Estimated angle")
plt.grid('True')
plt.legend()
plt.show()




#filtered_openpose_15 = lowpassfilter(openpose_trimmed[0:len(x_openpose_trimmed)], thresh=0.15)
#filtered_openpose_20 = lowpassfilter(openpose_trimmed[0:len(x_openpose_trimmed)], thresh=0.20)
filtered_openpose_25 = np.array(lowpassfilter(openpose_trimmed, thresh=0.25))
#filtered_openpose_30 = lowpassfilter(openpose_trimmed[0:len(x_openpose_trimmed)], thresh=0.30)
#filtered_openpose_35 = lowpassfilter(openpose_trimmed[0:len(x_openpose_trimmed)], thresh=0.35)


# Savgol Filtering
#filtered_openpose = signal.savgol_filter(openpose_trimmed[0:len(x_openpose_trimmed)], 51,2)

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

filtered_openpose_25 = filtered_openpose_25[:len(lemoh_trimmed)] #fix subtraction
# Calcula sinal de erro
err2 = lemoh_trimmed - filtered_openpose_25 
#err2 = np.resize(err2, (1, len(lemoh_trimmed)))
err2_plot = err2 + np.mean(lemoh_trimmed)

# Calcula SNR signal-to-noise ratio
mse2 = np.mean(err2 ** 2) # mean square error
signal_e2 = np.mean(lemoh_trimmed ** 2) # signal energy
SignalNR2 = 10 * np.log10(signal_e2/mse2)

print('The signal-noise ratio, after the filtering process, is: ', SignalNR2, ' dB')

aligned_filtered_openpose_25 = align_signals(lemoh_trimmed, filtered_openpose_25)


plt.figure()
plt.plot(x_lemoh_trimmed, lemoh_trimmed,'g', label="LEMOH")
plt.plot(x_lemoh_trimmed, aligned_filtered_openpose_25,'b', label="Openpose")
plt.plot(x_lemoh_trimmed, err2_plot, 'r', label="Error")
plt.title("Filtered Openpose")
plt.xlabel("Time (s)")
plt.ylabel("Estimated angle")
plt.grid('True')
plt.legend()
plt.show()


## CORRELATION MEASUREMENT 

arr2_interp = interpolate.interp1d(np.arange(aligned_filtered_openpose_25.size),aligned_filtered_openpose_25)
openpose2_stretch = arr2_interp(np.linspace(0,aligned_filtered_openpose_25.size-1,lemoh_trimmed.size))

# f = interpolate.interp1d(np.arange(0, len(filtered_openpose_25)), filtered_openpose_25)
# openpose_trimmed = f(np.linspace(0.0, len(filtered_openpose_25)-1, len(lemoh_trimmed)))

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
#plt.savefig('cloud_OpLh_Angle.eps', format='eps')
plt.show()

## calculating cross corelation and Pearson's corrcoef

Pear_coef = stats.pearsonr(np.array(np.squeeze(lemoh_trimmed)), np.array(np.squeeze(openpose2_stretch)))

# #Cross_coef = signal.correlate(op_right_hip_x_final, lb_right_hip_x_final)

plt.figure()
plt.plot(Pear_coef, color='red', linestyle='--')
plt.grid('True')
plt.title('Angle Pearson\'s Correlation = ' + "{:.4f}".format(Pear_coef[0]))
#plt.savefig('Pearsons_correlation_OpLh_Angle.eps', format='eps')
plt.show()


plt.figure()
x_corr = plt.xcorr(np.squeeze(lemoh_trimmed), np.squeeze(openpose2_stretch), usevlines=True, maxlags=100, normed=True, lw=2)
# #plt.xcorr(lb_right_hip_x_final, op_right_hip_x_final, usevlines=True, maxlags=58, normed=True, lw=2)
plt.grid('True')
plt.title('Angle Cross-correlation')
plt.xlabel('Lags')
plt.ylabel('Normalized Cross-correlation')
#plt.savefig('Cross_correlation_OpLh_Angle.eps', format='eps')
plt.show()

lags = x_corr[0]
c_values = x_corr[1]


print(Pear_coef)
print(a_fit,b_fit)
print('The lag value for the highest Xcorrelation is {}'.format(lags[np.argmax(c_values)]) + ', giving the {:.4f} score'.format(c_values[np.argmax(c_values)]))
