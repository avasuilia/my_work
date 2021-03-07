import numpy as np
hua=np.load('results/StarGAN_v2_trueemotions_gan-gp/ref_all.npz'
hops=192
import librosa
M=np.reshape(hua['norm_A'][4][0],(hops,hops))
ref_level_db=20
min_level_db=-100
def denormalize(S):
  return (((np.clip(S, -1, 1)+1.)/2.) * -min_level_db) + min_level_db
M=denormalize(M)+ref_level_db
M=librosa.db_to_power(M)

IM=librosa.feature.inverse.mel_to_audio(M, sr=16000, n_iter=2000,n_fft=6*hops,hop_length=hops,win_length=6*hops)
import soundfile as sf
sf.write('stereo_file1.wav', IM,16000)

import IPython
IPython.display.Audio('stereo_file1.wav')
