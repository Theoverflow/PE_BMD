#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.stats as stats
from scipy import signal
from pywt import swt, iswt, swt_max_level
from sklearn.mixture import GaussianMixture
import biosignalsnotebooks as bsnb
import math
import padasip as pa


Files = []
Datasets = []
for i in range(1,8):
    if i==1:
        Files.append(h5py.File('../Correlation_stress_datasets/1st_stress.h5', 'r'))
    elif i==2:
        Files.append(h5py.File('../Correlation_stress_datasets/2nd_stress.h5', 'r'))
    elif i==3:
        Files.append(h5py.File('../Correlation_stress_datasets/3rd_stress.h5', 'r'))
    else:
        Files.append(h5py.File(f'../Correlation_stress_datasets/{i}th_stress.h5', 'r'))
for el in Files:
    Datasets.append(el['00:07:80:0F:80:1A']['raw'])

#Header du fichier 1st_stress.h5 qui donnne des informations supplémentaires
h1 = {"00:07:80:0F:80:1A": {"position": 0, "device": "biosignalsplux", "device name": "00:07:80:0F:80:1A", "device connection": "/dev/tty.biosignalsplux-Bluetoot", "sampling rate": 1000, "resolution": [16, 16, 16, 16], "firmware version": 775, "comments": "", "keywords": "", "mode": 0, "sync interval": 2, "date": "2020-12-14", "time": "12:53:59.523", "channels": [1, 2, 3, 4], "sensor": ["EGG", "EDA", "RESPIRATION", "XYZ"], "label": ["CH1", "CH2", "CH3", "CH4"], "column": ["nSeq", "DI", "CH1", "CH2", "CH3", "CH4"], "special": [{"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}], "digital IO": [0, 1], "sleeve color": ["", "", "", ""]}}


#%%
#Fonction qui récupère les données ECG, EDA, RR et ACC de Dataset afin de les plots dans la deuxième fonction. Le paramètre nous sert à définir la taille des données que l'on veut

def collectData(i):
    data_ecg, data_eda, data_rr, data_acc = [],[],[],[]
    for j in range(len(Datasets)):
        """ normalized_ecg = preprocessing.normalize(Datasets[j]['channel_1'][:i], norm="l2")
        normalized_ecg = preprocessing.normalize(Datasets[j]['channel_2'][:i], norm="l2")
        normalized_rr = preprocessing.normalize(Datasets[j]['channel_3'][:i], norm="l2")
        normalized_acc = preprocessing.normalize(Datasets[j]['channel_4'][:i], norm="l2") """
        ecg = Datasets[j]['channel_1'][:i]
        eda = Datasets[j]['channel_2'][:i]
        rr = Datasets[j]['channel_3'][:i]
        acc = Datasets[j]['channel_4'][:i]
        data_ecg.append(ecg)
        data_eda.append(eda)
        data_rr.append(rr)
        data_acc.append(acc)
    return [data_ecg, data_eda, data_rr, data_acc]

dataplot = collectData(-1)
#%%
#Fonction pour ploter les données, on sauvegarde 7 images rawdatasX.png comprenant 4 graphiques différents
def plotDatas():
    for j in range(len(Datasets)):
        i = 1
        label = ['ECG', 'EDA', 'RR', 'ACC']
        fig = plt.figure(figsize=(60,20))
        for el in dataplot:
            tm = np.linspace(0, len(el[j])//1000, len(el[j]))
            el[j] = el[j]/max(el[j])
            plt.subplot(4,1,i)
            plt.plot(tm, el[j], label=label[i-1])
            plt.title(f'Raw {label[i-1]} Acquisition')
            plt.xlabel('Acquistition')
            plt.ylabel(f'{label[i-1]} Value')
            plt.legend()
            i += 1
        plt.show()
        fig.savefig(f'../Plot/rawdatas{j+1}.png')

fig = plt.figure(figsize=(60,20), facecolor="white")
nbi = 1
for el in dataplot[0]:
    signal_mv = (((el/ 2**16) - 0.5) * 3000) / 1000
    tm = np.linspace(0, len(signal_mv)//1000, len(signal_mv))
    ax = fig.add_subplot(7,1,nbi)
    ax.set_title(f'Raw ECG Acquisition {nbi}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'ECG Signal (mV) {nbi}')
    plt.plot(tm, signal_mv)
    #ax.legend()
    nbi += 1
plt.show()
fig.savefig(f'../Plot/Signals/ECG/ECG.png', facecolor="white")
#plotDatas()

#%%

#Fonction pour déterminer le SNR de l'ECG
def findSNR():
    raw_signals = dataplot[0]
    sampling_rate_ecg = 1000
    for el in raw_signals:
        time_ecg = np.linspace(0, len(el)//1000, len(el))
        clean_signal = (((el/ 2**16) - 0.5) * 3000) / 1000
        vpp_signal_ecg = np.ptp(clean_signal)
        time_r_peaks, amplitude_r_peaks = bsnb.detect_r_peaks(np.ravel(clean_signal), sampling_rate_ecg, time_units=True, plot_result= False)
        # Etape a faire : plot les signaux ainsi que amplitude_r_peaks pour visualiser les pic
        # Time of the onset of the sixth R peak (remember that Python lists and arrays start at 0)
        onset_sixth_hb = time_r_peaks[5]

        # Time of the onset of the fifth R peak
        onset_fifth_hb = time_r_peaks[4]

        # The start of the noise corresponds to the interval between peaks. Through observation, the fifth heartbeat ends at
        # around 0.5 s after its start, while the sixth starts at around 0.65 s after the start of the previous
        time_start_noise = onset_sixth_hb + 0.5
        time_end_noise = onset_sixth_hb + 0.65

        # Then, we need to convert it to index to identify it in the signal. The values are cast to integers because all indexes are integers.
        start_noise = int(time_start_noise * sampling_rate_ecg)
        end_noise = int(time_end_noise * sampling_rate_ecg)

        # Now we will identify the heartbeats. The procedure is analogous, and so we will do it in single lines.
        # The parcels of 0.7 and 1.5 correspond to emprical values that need to be added to the R peaks in order
        # to identify the onset of the heartbeats, once they do not start with the R peaks.
        time_start = time_ecg[int((onset_fifth_hb + .7)*sampling_rate_ecg):start_noise]
        beat_start = signal_mv[int((onset_fifth_hb+ .7)*sampling_rate_ecg):start_noise]

        time_end = time_ecg[end_noise:int((onset_sixth_hb + 1.5)*sampling_rate_ecg)]
        beat_end = signal_mv[end_noise:int((onset_sixth_hb + 1.5)*sampling_rate_ecg)]

        # signal with noise values
        time_noise = time_ecg[start_noise:end_noise]
        beat_noise = signal_mv[start_noise:end_noise]

        vpp_noise_ecg = []

        # For this task, we will follow the same procedure as shown before, but store the values in a list, so that we can then calculate the mean value.
        for t in time_r_peaks:
            start = int((t + 0.5) * sampling_rate_ecg) # 0.5 - time between a peak and a flat 
            end = int((t + 0.65)* sampling_rate_ecg) # 0.65 time between a peak and the end of the flat
            interval = signal_mv[start:end]
            vpp = np.ptp(interval)
            vpp_noise_ecg.append(vpp)
            
        vpp_noise_ecg = np.mean(vpp_noise_ecg)

        snr_ecg = vpp_signal_ecg/vpp_noise_ecg
        # The multiplication by 20 is because the signals are in the unit of (micro)Siemes
        snr_ecg_db = 20 * math.log10(snr_ecg)

        print("SNR for ECG signal: {}".format(snr_ecg))
        print("SNR for ECG signal: {} dB".format(snr_ecg_db))
    print("\n\n")
findSNR()

#%%
from scipy.integrate import simps
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import save

def ECGMetrics(sig, nb):
    sig = np.array(sig).flatten()
    output_file(f"../Plot/Signals/ECG/Metrics{nb}.html")
    
    Fs = 1000
    timee = np.linspace(0,len(sig)//Fs, len(sig))
    tachogram_data, tachogram_time = bsnb.tachogram(sig, Fs, signal=True, out_seconds=True)
    f1 = bsnb.plot(tachogram_time, tachogram_data, x_axis_label='Time (s)', y_axis_label='Cardiac Cycle (s)', title="Tachogram",  x_range=(0, timee[-1]))
    tachogram_data_NN, tachogram_time_NN = bsnb.remove_ectopy(tachogram_data, tachogram_time)
    bpm_data = (1 / np.array(tachogram_data_NN)) * 60
    f2 = bsnb.plot_post_ecto_rem_tachogram(tachogram_time, tachogram_data, tachogram_time_NN, tachogram_data_NN)
    # Maximum, Minimum and Average RR Interval
    max_rr = max(tachogram_data_NN)
    min_rr = min(tachogram_data_NN)
    avg_rr = np.average(tachogram_data_NN)

    # Maximum, Minimum and Average Heart Rate
    max_hr = 1 / min_rr # Cycles per second
    max_bpm = max_hr * 60 # BPM

    min_hr = 1 / max_rr # Cycles per second
    min_bpm = min_hr * 60 # BPM

    avg_hr = 1 / avg_rr # Cyles per second
    avg_bpm = avg_hr * 60 # BPM

    # SDNN
    sdnn = np.std(tachogram_data_NN)

    time_param_dict = {"Maximum RR": max_rr, "Minimum RR": min_rr, "Average RR": avg_rr, "Maximum BPM": max_bpm, "Minimum BPM": min_bpm, "Average BPM": avg_bpm, "SDNN": sdnn}
    f3 = bsnb.plot_hrv_parameters(tachogram_time, tachogram_data, time_param_dict)

    # Auxiliary Structures
    tachogram_diff = np.diff(tachogram_data)
    tachogram_diff_abs = np.fabs(tachogram_diff)
    sdsd = np.std(tachogram_diff)
    rr_i = tachogram_data[:-1]
    rr_i_plus_1 = tachogram_data[1:]
    
    # PoincarÃ© Parameters
    sd1 = np.sqrt(0.5 * np.power(sdsd, 2))
    sd2 = np.sqrt(2 * np.power(sdnn, 2) - np.power(sd1, 2))
    sd1_sd2 = sd1 / sd2
    f4 = bsnb.plot_poincare(tachogram_data)
    
    # Auxiliary Structures
    freqs, power_spect = bsnb.psd(tachogram_time, tachogram_data) # Power spectrum.

    # Frequemcy Parameters
    freq_bands = {"ulf_band": [0.00, 0.003], "vlf_band": [0.003, 0.04], "lf_band": [0.04, 0.15], "hf_band": [0.15, 0.40]}
    power_values = {}
    total_power = 0

    band_keys = freq_bands.keys()
    for band in band_keys:
        freq_band = freq_bands[band]
        freq_samples_inside_band = [freq for freq in freqs if freq >= freq_band[0] and freq <= freq_band[1]]
        power_samples_inside_band = [p for p, freq in zip(power_spect, freqs) if freq >= freq_band[0] and freq <= freq_band[1]]
        power = round(simps(power_samples_inside_band, freq_samples_inside_band), 5)
        
        # Storage of power inside each band
        power_values[band] = power
        
        # Total power update
        total_power = total_power + power
    
    f5 = bsnb.plot_hrv_power_bands(freqs, power_spect)

    grid = gridplot([[f1],[f2],[f3],[f4],[f5]])
    show(grid)

ECGMetrics(dataplot[0][6], 7)

#%%
# Fonction pour déterminer les pics R


def ECGParameters(nbecg):
    raw_signals = dataplot[0][nbecg]
    raw_signals = raw_signals/max(raw_signals)
    accelerometersData = dataplot[3][nbecg]
    accelerometersData = accelerometersData/max(accelerometersData)
    timee = np.linspace(0, len(raw_signals)//1000, len(raw_signals))
    f = pa.filters.FilterLMS(1, mu=0.01, w="zeros")
    sigECG, errECG, wECG = f.run(raw_signals,accelerometersData)
    sr = 1000
    """ fig = plt.figure(figsize=(60,40), facecolor='white')
    plt.plot(timee[150000:200000], raw_signals[150000:200000], 'b-', label=f'Raw ECG {nbecg}')
    plt.plot(timee[150000:200000], sigECG[150000:200000], 'r-', label=f'filtered ECG {nbecg}')
    plt.legend()
    plt.show()
    fig.savefig(f'../Plot/Signals/ECG/adaptivefilteringECG{nbecg}.png', facecolor="white") """
    dictParameters = bsnb.hrv_parameters(np.ravel(sigECG), sr, signal=True) 
    file = open("../ECG_Metrics.txt", 'w') 
    file.write(f'ECG {nbdata+1} Parameter : \n {dictParameters} \n')
    file.close()

#for i in range(len(dataplot[0])):
ECGParameters(3)

#%%
# Cette fonction est basée sur l'article suivant : Signal Processing Techniques for Removing Noise from ECG Signals
# On construit les filtres présentés dans cet article pour retirer le bruit provenant de la baseline, de la powerline, des muscles, ou du mouvements

 
def filtECG(N_order = 2):

    sample_frequence = 1000.    #Fréquence d'échantillonnage
    res = 16                    #Résolution de l'ADC
    vcc=3                       #Alimentation 
    fpLP = [0.35, 0.5, 5, 35]             #Fréquences de coupure
    fpBP = [[0.016, 5],[0.05, 35],[0.045, 0.25],[0.0167, 0.25]]
    nyquist_frequence = sample_frequence/2.
    ecg_signals = dataplot[0]
    


    for flp in fpLP:
        fig = plt.figure(figsize=(60,40), facecolor="white")
        b, a = signal.butter(N_order, flp, 'low', fs = sample_frequence)
        w, h = signal.freqs(b, a)
        ifilt = 1
        plt.subplot(8,1,ifilt)
        ifilt += 1
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title(f'Butterworth {N_order} order filter with frequency cutoff at cutoff {flp} Hz')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.axvline(flp, color='green')
        for el in ecg_signals:
            signal_us = ((el / 2**res) * vcc) / 0.12 # On transforme le signal pour lui donner une dimension (formule trouvée sur biosignalplus)
            filteredel = signal.filtfilt(b, a, np.ravel(signal_us)) #Application du premier filtre sur le signal
            time = np.linspace(0, len(el)//1000, len(el))
            plt.subplot(8,1,ifilt)
            plt.plot(time, signal_us, label='Raw ECG')
            plt.plot(time, filteredel, 'r-', label=f'Low-pass filtered ECG at {flp} Hz')
            plt.title(f'Filtered ECG {ifilt-1}')
            plt.xlabel('Time (s)')
            plt.ylabel(f'ECG {ifilt} (uS)')
            plt.legend()
            ifilt += 1
        plt.show()
        # fig.savefig(f'../filteredECGLP_{N_order}_{flp}.png', facecolor="white")
        plt.close(fig)
    
    for fbp in fpBP:
        fig = plt.figure(figsize=(60,40), facecolor="white")
        b, a = signal.butter(N_order, fbp, 'bandpass', fs = sample_frequence)
        w, h = signal.freqs(b, a)
        ifilt = 1
        plt.subplot(8,1,ifilt)
        ifilt += 1
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title(f'Butterworth {N_order} order  BPfilter with frequency cutoff at cutoff [{fbp[0]}-{fbp[1]}] Hz')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.axvline(fbp[0], color='green')
        plt.axvline(fbp[1], color='green')
        for el in ecg_signals:
            signal_us = ((el / 2**res) * vcc) / 0.12 # On transforme le signal pour lui donner une dimension (formule trouvée sur biosignalplus)
            filteredel = signal.filtfilt(b, a, np.ravel(signal_us)) #Application du premier filtre sur le signal
            time = np.linspace(0, len(el)//1000, len(el))
            plt.subplot(8,1,ifilt)
            plt.plot(time, signal_us, label='Raw ECG')
            plt.plot(time, filteredel, 'r-', label=f'Band-pass filtered ECG at [{fbp[0]}-{fbp[1]}] Hz')
            plt.title(f'Filtered ECG {ifilt-1}')
            plt.xlabel('Time (s)')
            plt.ylabel(f'ECG {ifilt} (uS)')
            plt.legend()
            ifilt += 1
        plt.show()
        ifilt = 1 
        #fig.savefig(f'../filteredecgBP_{N_order}_{fbp}.png', facecolor="white") 
        plt.close(fig)

    
    Q = 30
    fnotch = [30, 40, 50]
    for fn in fnotch:
        fig = plt.figure(figsize=(60,40), facecolor="white")
        b, a = signal.iirnotch(fn, Q, sample_frequence)
        w, h = signal.freqz(b, a)
        ifilt = 1
        plt.subplot(8,1,ifilt)
        ifilt += 1
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title(f'Butterworth {N_order} order  BPfilter with frequency cutoff at cutoff {fn} Hz')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.axvline(fn, color='green')
        for el in ecg_signals:
            signal_us = ((el / 2**res) * vcc) / 0.12 # On transforme le signal pour lui donner une dimension (formule trouvée sur biosignalplus)
            filteredel = signal.filtfilt(b, a, np.ravel(signal_us)) #Application du premier filtre sur le signal
            time = np.linspace(0, len(el)//1000, len(el))
            plt.subplot(8,1,ifilt)
            plt.plot(time, signal_us, label='Raw ECG')
            plt.plot(time, filteredel, 'r-', label=f'Band-pass filtered ECG at {fn} Hz')
            plt.title(f'Filtered ECG {ifilt-1}')
            plt.xlabel('Time (s)')
            plt.ylabel(f'ECG {ifilt} (uS)')
            plt.legend()
            ifilt += 1
        plt.show()
        ifilt = 1 
        #fig.savefig(f'../Plot/Signals/ECG/filteredecgNOTCH_{N_order}_{fn}.png', facecolor="white") 
        plt.close(fig)


for j in range(1, 5):
    filtECG(j)