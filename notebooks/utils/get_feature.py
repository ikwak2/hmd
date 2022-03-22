import soundfile as sf
import numpy as np
import librosa


def feature_extract_melspec(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100):

    data, sample_rate = librosa.load(fnm, sr = 4000)
    data = data * 1.0

    if samp_sec:
        if len(data) > sample_rate * samp_sec :
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i]

        Sig.append(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length)))

    return Sig


def feature_extract_stft(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, hop_length=256, win_length = 512, n_mels = 100):

    data, sample_rate = librosa.load(fnm, sr = sr)
    data = data * 1.0

    if samp_sec:
        if len(data) > sample_rate * samp_sec :
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i]

#        D_highres = librosa.stft(emphasized_signal, hop_length=hop_length, n_fft=win_length)
#        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
            
        Sig.append(librosa.power_to_db(np.abs(librosa.stft(emphasized_signal, n_fft=win_length, hop_length=hop_length, win_length=win_length))))

    return Sig


def feature_extract_cqt(fnm, samp_sec=20, sr = 4000, pre_emphasis = 0, filter_scale = 1, n_bins = 80, fmin = 10):

    data, sample_rate = sf.read(fnm, dtype = 'int16')
    #data, sample_rate = librosa.load(fnm, sr = sr)
    data = data * 1.0

    if samp_sec:
        if len(data) > sample_rate * samp_sec :
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))])
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]
    else:
        n_samp = 1
        signal = [data]

    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i]

#        D_highres = librosa.stft(emphasized_signal, hop_length=hop_length, n_fft=win_length)
#        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        Sig.append(np.log(np.abs(librosa.cqt(emphasized_signal, sr=sample_rate,
                                             filter_scale = filter_scale, n_bins=n_bins,
                                             fmin = fmin))+1) )

    return Sig
