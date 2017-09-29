import numpy as np
from scipy.signal import spectrogram
from scipy.io.wavfile import read as read_wav
from PyQt5.QtGui import QImage
import warnings

def linear_to_logarithmic(spectrum, frequencies, frequencies_log=None):
    df = frequencies[1]-frequencies[0]
    M = min(len(frequencies), len(spectrum))
    f_0 = frequencies[0]
    a = (frequencies[M-1]/f_0)**(1/(M-1))
    if frequencies_log == None: # don't calculate this every time
        frequencies_log = [round(f_0*a**i,2) for i in range(M)]
    frequencies_k = [(frequencies_log[i] - f_0)/df for i in range(M)]
    spec_log = list()
    for i in range(len(frequencies)):
        k = frequencies_k[i]
        k0 = int(k)
        k1 = int(k+1)
        a = spectrum[k0]
        if k1 < len(spectrum):
            a += (spectrum[k1]-spectrum[k0])*(k-k0)
        spec_log.append(a)
    return spec_log, frequencies_log
    
def spectrogram_to_logarithmic(spectrogram, frequencies):
    freq_lin = frequencies
    freq_log = None
    Sxx_lin = spectrogram
    Sxx_log_ = list()
    for spec_lin_raw in np.transpose(Sxx_lin):
        spec_lin = list(spec_lin_raw[1:])
        spec_log, freq_log = linear_to_logarithmic(
            spec_lin, freq_lin, freq_log)
        Sxx_log_.append(spec_log)
    Sxx_log = [list(x) for x in zip(*Sxx_log_)]
    return Sxx_log, freq_log
    
def wav_to_png(inputname, outputname):
    print("[%s] reading..." % inputname)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samplerate, samples = read_wav(inputname)
    except:
        print("Could not read file: `%s'."%inputname)
        return False
    print("[%s] calculating spectrogram..." % inputname)
    f, t, Sxx_lin = spectrogram(
            samples,
            fs=samplerate,
            nperseg=2048,
            window="hamming"
        )
    ma = np.min(Sxx_lin[Sxx_lin>1e-10])
    Sxx_lin[Sxx_lin<1e-9] = ma
    Sxx_lin = np.log2(Sxx_lin/ma)**4
    print("[%s] logscaling..." % inputname)
    Sxx_log, freq_log = spectrogram_to_logarithmic(Sxx_lin, f[5:300])
    print("[%s] plotting..." % inputname)
    height, width = np.shape(Sxx_log)
    image = QImage(width, height, QImage.Format_RGB32)
    normvalue = 255/np.max(Sxx_log)
    for y in range(height):
        for x in range(width):
            v = int(Sxx_log[y][x]*normvalue)
            image.setPixel(x, height-y-1, v + (v<<8) + (v<<16))
    image.save(outputname)

def main(*argv):
    if len(argv) < 2:
        print("Usage: %s inputfile outputfile"%argv[0])
        return False
    inputname = argv[1]
    outputname = argv[2]
    wav_to_png(inputname, outputname)
    
if __name__ == "__main__":
    from sys import argv
    main(*argv)
