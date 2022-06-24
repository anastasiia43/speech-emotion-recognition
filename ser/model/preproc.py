import math
import os
import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import librosa.display
import numpy as np
import os

import noisereduce as nr
from os import path
from pydub import AudioSegment


def convert_mp3_to_wav(src):
    dst = "E:\paper_work\project\ser\media\documents\\test.wav"
    # convert wav to mp3
    try:
        audio = AudioSegment.from_file(src, "mp3")
    except:
        audio = AudioSegment.from_file(src, format="mp4")
    audio.export(dst, format="wav")
    return dst

def load_audio(path):
    path_wav = convert_mp3_to_wav(path)
    audio = librosa.load(path_wav, res_type='kaiser_fast', sr=44000)

    #remove file
    os.remove(path)
    os.remove(path_wav)


    new_audio = audio[0]

    length_chosen = 120378
    if new_audio.shape[0] < 300000:
        if new_audio.shape[0] > length_chosen:
            new_audio = new_audio[:length_chosen]
        if new_audio.shape[0] < length_chosen:
            new_audio = new_audio[:length_chosen]
        elif new_audio.shape[0] > length_chosen:
            new_audio = np.pad(new_audio, math.ceil((length_chosen-new_audio.shape[0])/2), mode='median')
    return np.array(new_audio)

def get_feature(audio):
    final_x = nr.reduce_noise(audio, sr=44000) #updated 03/03/22

    rms = librosa.feature.rms(final_x, frame_length=2048, hop_length=512).T # Energy - Root Mean Square
    zcr = librosa.feature.zero_crossing_rate(final_x , frame_length=2048, hop_length=512, center=True).T # ZCR
    mfccs = librosa.feature.mfcc(y=final_x, sr=44000, n_mfcc=40).T

    f_rms = np.asarray(rms).astype('float32')
    f_zcr = np.asarray(zcr).astype('float32')
    f_mfccs = np.asarray(mfccs).astype('float32')

    mfccs = np.concatenate((f_zcr, f_rms, f_mfccs), axis=1)
    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(1, 236, 42)
    return mfccs
