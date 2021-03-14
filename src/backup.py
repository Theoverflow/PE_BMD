#Importation des librairies utiles pour les étapes de traitement

import numpy as np
import h5py
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_filter
import cvxopt as cv
import cvxopt.solvers
import scipy.stats as stats
from scipy.stats import norm
from scipy import signal
from pywt import swt, iswt, swt_max_level
from sklearn.mixture import GaussianMixture
from copy import deepcopy

#Dans Files on met l'ensemble des données des 7 expériences
#Dans Datasets on mets l'ensemble des données d'acquisitions (ECG, EDA, Respiration Rate et Acceleration)

Files = []
Datasets = []
for i in range(1,8):
    if i==1:
        Files.append(h5py.File('..\Correlation_stress_datasets\1st_stress.h5', 'r'))
    elif i==2:
        Files.append(h5py.File('..\Correlation_stress_datasets\2nd_stress.h5', 'r'))
    elif i==3:
        Files.append(h5py.File('../Correlation_stress_datasets/3rd_stress.h5', 'r'))
    else:
        Files.append(h5py.File(f'../Correlation_stress_datasets/{i}th_stress.h5', 'r'))
for el in Files:
    Datasets.append(el['00:07:80:0F:80:1A']['raw'])

#Header du fichier 1st_stress.h5 qui donnne des informations supplémentaires
h1 = {"00:07:80:0F:80:1A": {"position": 0, "device": "biosignalsplux", "device name": "00:07:80:0F:80:1A", "device connection": "/dev/tty.biosignalsplux-Bluetoot", "sampling rate": 1000, "resolution": [16, 16, 16, 16], "firmware version": 775, "comments": "", "keywords": "", "mode": 0, "sync interval": 2, "date": "2020-12-14", "time": "12:53:59.523", "channels": [1, 2, 3, 4], "sensor": ["EGG", "EDA", "RESPIRATION", "XYZ"], "label": ["CH1", "CH2", "CH3", "CH4"], "column": ["nSeq", "DI", "CH1", "CH2", "CH3", "CH4"], "special": [{"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}, {"SpO2": [80, 40, 5, 4, 3, 3]}], "digital IO": [0, 1], "sleeve color": ["", "", "", ""]}}
#Deux fonctions permettants d'afficher lla structures des données. Elles sont similaires mais comme on ne peut pas distinguer un Groupe HDF d'un Dataset, on doit adapter les scripts.
def afficheLabels(dset1):
    for key in dset1.keys():
        print(key)
        print(f'{list(dset1[key].keys())}')
        for subkey in dset1[key].keys():
            print(f'{dset1[key][subkey]} \n')
#afficheLabels()

def afficheLabelsSupport(dset1):
    l = 'support'
    for key in dset1['support'].keys():
        print(key)
        print(f'{dset1[l][key].keys()} \n')
        for subkey in dset1[l][key].keys():
            print(f'{dset1[l][key][subkey]} \n')
            for subsubkey in dset1[l][key][subkey].keys():
                print(f'{dset1[l][key][subkey][subsubkey]} \n')
#afficheLabelsSupport()
data = Files[0]['00:07:80:0F:80:1A']['support']['level_1000']['channel_1']['Mx']

"""
# Architecture des données
Les fichiers x_stress.h5 sont organisé de la manière suivante
* Le nom de l'appareil '00:07:80:0F:80:1A'
    * digital : 
        * digital_1 : Grande liste de 0
    * events
        * digital : Un tableau [2,0,0,0] correspondant à la durée de l'intervalle de synchronisation
        * sync : Un tableau vide
    * plugin : Un groupe avec des Datasets vides 
    * raw :
        * channel_1 : Un tableau de N entrées correspondant aux valeurs de l'ECG
        * channel_2 : Un tableau de N entrées correspondant aux valeurs de l'EDA
        * channel_3 : Un tableau de N entrées correspondant aux valeurs du taux de respiration
        * channel_4 : Un tableau de N entrées correspondant aux valeurs de l'accéleromètre
        * nSeq : Un tableau de 0 à 330599 correspondant à l'échantillonnage
    * support : 
        * level_10 :
            * channel_1 : 
                * Mx : 
                * mean : 
                * mean_x2 : 
                * mx : 
                * sd :
                * t : 
            * channel_2 : Idem
            * channel_3 : Idem
            * channel_4 : Idem
            * dig_channel_1 :
        * level_100 : Idem avec une taille de 3306
        * level_1000 : Idem avec une taille de 330

# Conclusion
Les données à traiter se trouvent dans '/00:07:80:0F:80:1A/raw' et '/00:07:80:0F:80:1A/support/level_x' qu'ils faut examiner
"""

#Fonction qui récupère les données ECG, EDA, RR et ACC de Dataset afin de les plots dans la deuxième fonction. Le paramètre nous sert à définir la taille des données que l'on veut
from sklearn import preprocessing

def collectData(i):
    data_ecg, data_eda, data_rr, data_acc = [], [],[],[]
    for j in range(len(Datasets)):
        """ normalized_ecg = preprocessing.normalize(Datasets[j]['channel_1'][:i], norm="l2")
        normalized_eda = preprocessing.normalize(Datasets[j]['channel_2'][:i], norm="l2")
        normalized_rr = preprocessing.normalize(Datasets[j]['channel_3'][:i], norm="l2")
        normalized_acc = preprocessing.normalize(Datasets[j]['channel_4'][:i], norm="l2") """
        normalized_ecg = Datasets[j]['channel_1'][:i]
        normalized_eda = Datasets[j]['channel_2'][:i]
        normalized_rr = Datasets[j]['channel_3'][:i]
        normalized_acc = Datasets[j]['channel_4'][:i]
        data_ecg.append(normalized_ecg)
        data_eda.append(normalized_eda)
        data_rr.append(normalized_rr)
        data_acc.append(normalized_acc)
    return [data_ecg, data_eda, data_rr, data_acc]

dataplot = collectData(-1)

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
            plt.plot(tm, el[j])
            plt.title(f'Raw {label[i-1]} Acquisition')
            plt.xlabel('Acquistition')
            plt.ylabel(f'{label[i-1]} Value')
            plt.legend()
            i += 1
        plt.show()
        fig.savefig(f'..\Plot\rawdatas{j+1}.png')

#plotDatas()

# Filtre passe-bas pour l'EDA. La fonction reçoit en argument l'ordre du filtre à créer qui est de 2 par défaut et trace le
# On trace d'abord les filtres pour les différentes fréquences de coupure voulues, puis on trace les signaux EDA correspondant aux 7 expériences réalisées. Les fréquences de coupure sont par défaut 0.35 Hz et 35 Hz et proviennent de la littérature.

def FiltEDA(N_order = 2):

    sample_frequence = 1000.    #Fréquence d'échantillonnage
    res = 16                    #Résolution de l'ADC
    vcc=3                       #Alimentation 
    fp = [0.35, 35]             #Fréquences de coupure
    nyquist_frequence = sample_frequence/2.
    fig = plt.figure(figsize=(60,40), facecolor="white")
    i = 3

#On génère un premier filtre
    b, a = signal.butter(N_order, fp[0], 'low', fs = sample_frequence)
    w, h = signal.freqs(b, a)
    plt.subplot(9,1,1)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title(f'Butterworth filter frequency response at cutoff {fp[0]} Hz')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.axvline(fp[0], color='green') 

#On génère le second filtre
    b1, a1 = signal.butter(N_order, fp[1], 'low', fs = sample_frequence)
    w1, h1 = signal.freqs(b1, a1)
    plt.subplot(9,1,2)
    plt.semilogx(w1, 20 * np.log10(abs(h1)))
    plt.title(f'Butterworth filter frequency response at  cutoff {fp[1]} Hz')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(fp[1], color='green')

    for el in dataplot[1]:
        signal_us = ((el / 2**res) * vcc) / 0.12 # On transforme le signal pour lui donner une dimension (formule trouvée sur biosignalplus)
        filteredel = signal.filtfilt(b, a, np.ravel(signal_us)) #Application du premier filtre sur le signal
        filteredel1 = signal.filtfilt(b1, a1, np.ravel(signal_us)) #Application du second filtre sur le signal
        time = np.linspace(0, len(el)//1000, len(el))
        plt.subplot(9,1,i)
        plt.plot(time, signal_us, label='Raw EDA')
        plt.plot(time, filteredel, 'r-', label='Low-pass filtered EDA at 0.35Hz')
        plt.plot(time, filteredel1, 'g-', label='Low-pass filtered EDA at 35Hz')
        plt.title(f'Filtered EDA {i-2}')
        plt.xlabel('Time')
        plt.ylabel(f'EDA {i} (uS)')
        plt.legend()
        i += 1
    plt.show()
    fig.savefig(f'..\Plot\Signals\EDA\filterededa_{N_order}_{fp[0]}_{fp[1]}.png', facecolor="white")

for j in range(1, 9):
    FiltEDA(j)


# Smooth function 

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError ("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

#import biosignalsnotebooks as bsnb
# Fonction traitant le signal EDA en suivant le tuto du site biosignalplus du constructeur de l'outil de test
# On filtre avec un passe bas d'ordre 4 à 0.35Hz 

def SWTLevel(nb):
    a = nb//64
    return a*64

res = 16
vcc = 3
def processEDA():
    fig = plt.figure(figsize=(60,40), facecolor="white")
    b, a = signal.butter(4, 0.35, 'low', fs = 1000.)
    ligne = 1
    processedEDA = []
    for el in dataplot[1]:
        signal_us = ((el / 2**res) * vcc) / 0.12
        signal_us_low_pass = signal.filtfilt(b, a, np.ravel(signal_us))
        N = len(signal_us_low_pass)

        swtN = SWTLevel(N) #On détermine la taille maximale qu'on peut avoir pour qu'elle soit un multiple du carré du niveau par défaut (niveau 8 donc 64)
        lvl = swt_max_level(swtN) #Calcul du niveaux max qu'on peut obtenir avec la taille swtN
        print(lvl)
        swt_orig_coeffs = swt(signal_us_low_pass[:swtN], "haar", level=lvl) #Application ondelette de Haar
        detail_coeffs = swt_orig_coeffs[0][1]
        scaling_coeffs = swt_orig_coeffs[0][0]
        time = np.linspace(0, N//1000, N)
        time1 = np.linspace(0, swtN//1000, swtN)

        #Generation of a Gaussian Mixture model. "One Gaussian component describes coefficients centered around zero, and the other describes those spread out at larger values... The Gaussian with smaller variance corresponds to the wavelet coefficients of Skin Conductance Level (SCL) , while the Gaussian with larger variance corresponds to the wavelet coefficients of Skin Conductance Responses (SCRs) "
        gaussian_mixt = GaussianMixture(n_components=2, covariance_type="spherical")
        detail_coeffs_col = np.reshape(detail_coeffs, (len(detail_coeffs), 1))
        gaussian_mixt.fit(detail_coeffs_col)

        #Determination of the Cumulative Density Function ( Φmixt ) of the previously defined Gaussian Mixture Φmixt=weight1×Φ1+weight2×Φ2
        norm_1 = norm(loc=gaussian_mixt.means_[0][0], scale=np.sqrt(gaussian_mixt.covariances_[0])) 
        norm_2 = norm(loc=gaussian_mixt.means_[1][0], scale=np.sqrt(gaussian_mixt.covariances_[1])) 
        weight_1 = gaussian_mixt.weights_[0]
        weight_2 = gaussian_mixt.weights_[1]
        sort_detail_coeffs = np.sort(detail_coeffs)
        norm_1_cdf = norm_1.cdf(sort_detail_coeffs)
        norm_2_cdf = norm_2.cdf(sort_detail_coeffs)
        cdf_mixt = weight_1 * norm_1_cdf + weight_2 * norm_2_cdf

        #  Definition of motion artifact removal thresholds using the Cumulative Distribution Function (CDF) of the previously defined Gaussian Mixture model, considering an artifact proportion value  δ  equal to 0.01
        art_prop = 0.01 # Artifact proportion value.
        low_thr = None 
        high_thr = None
        # Check when the CDF mixture function reaches values art_prop / 2 and 1 - art_prop / 2.
        for i in range(0, len(norm_1_cdf)):
            # Low threshold clause.
            if cdf_mixt[i] - cdf_mixt[0] >= art_prop and low_thr == None:
                low_thr = sort_detail_coeffs[i]
            # High threshold clause.
            if cdf_mixt[-1] - cdf_mixt[i] <= art_prop and high_thr == None:
                high_thr = sort_detail_coeffs[i]
        
        #Removal of wavelet coefficients related with motion artifacts
        filt_detail_coeffs = deepcopy(detail_coeffs)
        count_1 = 0
        count_2 = 0
        for j in range(0, len(filt_detail_coeffs)):
            if detail_coeffs[j] <= low_thr or detail_coeffs[j] >= high_thr:
                filt_detail_coeffs[j] = 0
            else:
                continue
        # Update of the SWT decomposition tupple.
        sr = 1000
        swt_coeffs = [(np.array(scaling_coeffs), np.array(filt_detail_coeffs))]
        rec_signal = iswt(swt_coeffs, "haar")
        signal_int = smooth(rec_signal, sr * 3)
        signal_int = signal_int * (max(signal_us_low_pass) / max(signal_int))
        plt.subplot(9, 1, ligne)
        ligne += 1
        time2 = np.linspace(0, len(signal_int)//1000, len(signal_int))
        plt.plot(time2, signal_int)
        processedEDA.append(signal_int)
    plt.show()
    fig.savefig(f'..\Plot\Signals\EDA\processedEDA.png', facecolor="white")
    return processedEDA

processEDA()

def cvxEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol':1e-9}):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """

    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,y,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C], 
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*y,  -(Ct*y), -(Bt*y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n+nC]
    t = B*l + C*d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))

dataprocessed = processEDA()
def plotCVxEDA():
    for el in cvxEDA:
        y = np.array(el)
        #yn = stats.zscore(y)
        Fs = 1000
        [r, p, t, l, d, e, obj] = cvxEDA(dataprocessed, 1/Fs)
        tm = np.linspace(0, len(y)//1000, len(y))

        fig = plt.figure(figsize=(60,20), facecolor="white")
        plt.subplot(4, 1, 1)
        plt.plot(tm, y)
        plt.title('Processed EDA signal')
        plt.xlabel('Time')
        plt.ylabel('Normalized Smoothed EDA value')
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(tm, r)
        plt.title('Phasic component')
        plt.xlabel('Time')
        plt.ylabel('Phasic component value')
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(tm, p)
        plt.title('Sparse SMNA driver of phasic component')
        plt.xlabel('Time')
        plt.ylabel('Sparse SMNA value')
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(tm, t)
        plt.title('Tonic component')
        plt.xlabel('Time')
        plt.ylabel('Tonic component value')
        plt.legend()
        plt.show()
        fig.savefig(f'..\Plot\Signals\EDA\CVxEDA{i+1}.png', facecolor="white")

plotCVxEDA()

"""
# Explication de cvxEDA

Electrodermal Activity (EDA) représente l'activité électrique de la peau.

On la mesure à l'aide de la conductance, et ses signaux sont une manifestation de l'activité des glande sudoripare (agissant sur la sueur) innervée par le système nerveux autonome.

Lorsque les nerfs sudomoteurs stimulent la production de sueur, la conductivité mesurée à la surface de la peau se modifie en raison de la sécrétion de sueur et des variations de perméabilité ionique des membranes des glandes sudoripares.

La conductance mesurée y est considérée comme une somme de 3 signaux : 
 * Une composante phasique r représentant la réponse courte suite aux stimuli
 * Une composante tonique t incluant les dérives lentes de la ligne de base et des fluctuations spontanées
 * Un bruit d'origine Gaussien

En considérant le problème comme un problème d'optimisation convexe quadratique, on obtient la forme de ses différentes composantes ainsi que l'activité des nerfs neuromoteurs p
"""


def plotECG(start=None, stop=None):
    Fs = 1000.0
    fig = plt.figure(figsize=(60,20), facecolor="white")
    for i in range(len(dataplot[0])):
        if start or stop:
            ecg = np.array(dataplot[0][i][start:stop])
        else:
            ecg = np.array(dataplot[0][i])
        ecg2 = heartpy.filtering.filter_signal(np.ravel(ecg), cutoff = 0.01, sample_rate = Fs, filtertype = 'notch')
        ecg2 = stats.zscore(ecg2)
        tm = np.linspace(0, len(ecg)//1000, len(ecg))
        if start or stop:
            tm2 = np.linspace(start//1000, stop//1000, len(ecg2))
        else: 
            tm2 = np.linspace(0, len(ecg2)//1000, len(ecg2))
        plt.subplot(4, 2, i+1)
        """ plt.plot(tm, ecg)
        plt.title(f'ecg Acquisition')
        plt.xlabel('Acquistition')
        plt.ylabel(f'ecg Value')
        plt.legend()
        plt.subplot(2, 1, 2) """
        plt.plot(tm2, ecg2)
        plt.title(f'ecg2 Acquisition {i+1}')
        plt.xlabel('Acquistition')
        plt.ylabel(f'ecg2 Value {i+1}')
        plt.legend()
    plt.show()
    fig.savefig(f"data{start}_{stop}.png")
#plotECG()
#plotECG(30000,40000)
