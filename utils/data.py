from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import librosa
import wave




class MdbStemSynthDataset(Dataset):
    def __init__(self, wav_dir, annot_dir, sr=None, transform=None):

        self.wav_dir = wav_dir
        self.annot_dir = annot_dir
        self.sr = sr
        wav_count = sum(1 for file in os.listdir(wav_dir) if (file.endswith(".wav") and not file.startswith("._")))
        annot_count = sum(1 for file in os.listdir(annot_dir) if file.endswith(".csv"))

        if wav_count == annot_count:
            self.data_fnames = [os.path.splitext(file)[0]  for file in os.listdir(annot_dir) if file.endswith(".csv")]
        else:
            print("number of audio files and annotations do not match")


    def __len__(self):
        return len(self.data_fnames)


    def __getitem__(self,idx):
        

        audio, samplerate = librosa.load(os.path.join(self.wav_dir, self.data_fnames[idx])+".wav",self.sr)
        pitch_annotation= pd.read_csv(os.path.join(self.annot_dir, self.data_fnames[idx])+".csv",header=None).to_numpy()

        # print(wave.open(os.path.join(self.wav_dir, self.data_fnames[idx])+".wav").getframerate())

        sample = {'name': self.data_fnames[idx], 'audio': audio, 'samplerate': samplerate, 'pitch': pitch_annotation}
        
        return sample


if __name__ == "__main__":


    mdb_dataset = MdbStemSynthDataset("datasets/MDB-stem-synth/audio_stems","datasets/MDB-stem-synth/annotation_stems")
    print(mdb_dataset.data_fnames[0])
    dat = mdb_dataset[0]
    print(type(dat[0]))
    print(dat.shape)
    