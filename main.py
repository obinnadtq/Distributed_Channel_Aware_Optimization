import numpy as np
import matplotlib.pyplot as plt

forward_channel_SNR_1 = 12
measurement_SNR = 3
i_x_y_3dB = [0.770507886273119, 1.103030556942549, 1.3140182588244824, 1.4659336307444786]
i_x_z_3dB_sca = [0.746602335216394, 1.0578392708208102, 1.267904364016096, 1.426379645320874]
i_x_z_3dB_sib = [0.7362014227719886, 1.0639755235377795, 1.273382528016092, 1.425379645320874]
# sca1 = [0.6958248707684584, 1.0069208501818578, 1.2102112499725655, 1.360082436250467 ]
# sib1 = [0.6426701233003094, 0.9641143733661014, 1.154970132695412, 1.3311346767565226 ]
x = [1, 2, 3, 4]
#
measurement_SNR2 = 7
i_x_y_7dB = [1.2206517190780224, 1.5831585163533264, 1.7689483751264738, 1.8699353447047895]
i_x_z_7dB_sca = [1.156774261054006, 1.5283117174736343, 1.7066492092463967, 1.8295942136387582]
i_x_z_7dB_sib = [1.1585148241849417, 1.5218484677992234, 1.7165907801503346, 1.8295942136387582]
# sca2 = [1.0758203435965468, 1.4305742685630007, 1.634959019265187, 1.7626087934534633]
# sib2 = [1.000226404040064, 1.3398592687033248, 1.5379318272994402, 1.7192575885719106]




# forward_channel_SNR_2 = 15
# measurement_SNR = 3
sca1 = [0.7277234308923679, 1.0422936060318637, 1.2555497976394525, 1.3976480265740174]
sib1 = [0.7044734595659184, 1.0351576847580963, 1.2386572226791337, 1.3962491411931718]
#
# measurement_SNR2 = 7
sca2 = [1.1429937075407273, 1.49544701859295, 1.6752413773911772, 1.7956947598475015]
sib2 = [1.1030994494769315, 1.4505698143246296, 1.653462405638045, 1.781609035496853]


#plt.plot(x, sca1, 'x-', linewidth=2, label='I(X;Zbar)-CA')
#plt.plot(x, sib1, 'x-', linewidth=2, label='I(X;Zbar)-IB')
#plt.plot(x, i_x_y_3dB, '*-', linewidth=2, label='I(X;Y)')
#plt.plot(x, i_x_z_3dB_sca, '*-', linewidth=2, label='I(X;Z)')
#plt.plot(x, i_x_z_3dB_sib, '*-', linewidth=2, label='I(X;Z)')
plt.plot(x, sca2, '+-', linewidth=2, label='I(X;Zbar)-CA')
plt.plot(x, sib2, '*-', linewidth=2, label='I(X;Zbar)-IB')
plt.plot(x, i_x_y_7dB, '*-', linewidth=2, label='I(X;Y)')
#plt.plot(x, i_x_z_7dB_sca, '*-', linewidth=2, label='i_x_z_sca')
plt.plot(x, i_x_z_7dB_sib, '*-', linewidth=2, label='I(X;Z)')
plt.yticks(np.arange(0.6, 2, 0.2))
plt.xticks(np.arange(1, 5, 1))
plt.ylabel(r'I(X;$\bar{Z}$) bits')
plt.xlabel('Number of sensors')
plt.text(2, 1.4, r'$\gamma_{m} = 7 dB$, $\gamma_{c} = 15 dB$')
plt.grid()
plt.legend()
plt.title('Relevant information vs. Number of sensors')
plt.show()
