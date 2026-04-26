import wfdb
import os

def load_ecg(record_name):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    full_path = os.path.join(base_dir, record_name)

    record = wfdb.rdrecord(full_path)
    signal = record.p_signal[:, 0]
    fs = record.fs

    # try loading annotations (optional)
    try:
        annotation = wfdb.rdann(full_path, 'atr')
        r_peaks = annotation.sample
    except:
        r_peaks = None

    return signal, fs, r_peaks