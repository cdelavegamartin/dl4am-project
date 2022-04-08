from __future__ import print_function, division
from typing import Any, Dict
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import librosa




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


    def __getitem__(self,index):
        
        name = self.data_fnames[index]
        audio, samplerate = librosa.load(os.path.join(self.wav_dir, name)+".wav",sr=self.sr)
        pitch_annotation= pd.read_csv(os.path.join(self.annot_dir, name)+".csv",header=None).to_numpy()

        # print(wave.open(os.path.join(self.wav_dir, name)+".wav").getframerate())

        sample = {'name': name, 'audio': audio, 'samplerate': samplerate, 'pitch': pitch_annotation}
        
        return sample


class NsynthDataset(Dataset):
    def __init__(self, root_dir, split='test', sr=None, pitch_notation='hz', transform=None):

        self.root_dir = root_dir
        self.split = split
        self.sr = sr
        self.pitch_notation = pitch_notation
        self.split_dir = os.path.join(self.root_dir,f"nsynth-{split}")
        self.json_data = pd.read_json(os.path.join(self.split_dir,"examples.json")).T

    
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        
        sample_data = self.json_data.iloc[index]
        name = sample_data.note_str
        pitch_midi = sample_data.pitch
        if self.pitch_notation=='hz':
            pitch_hz = 2**((pitch_midi-69)/12)*440
            pitch_annotation = np.asarray([0.0,pitch_hz]).reshape(1,-1)
        else:
            print("pitch notation is not set or invalid for the dataset")

        audio, samplerate = librosa.load(os.path.join(self.split_dir, "audio", name+".wav"),sr=self.sr)
        sample = {'name': name, 'audio': audio, 'samplerate': samplerate, 'pitch': pitch_annotation}

        return sample




if __name__ == "__main__":

    dataset='nsynth'

    if dataset=='mdb':

        mdb_dataset = MdbStemSynthDataset("datasets/MDB-stem-synth/audio_stems","datasets/MDB-stem-synth/annotation_stems")
        
        dat = mdb_dataset[0]
        
        print(type(dat['pitch']), dat['pitch'].shape)
        print(type(dat['audio']), dat['audio'].shape)
        

    elif dataset=='nsynth':

        nsynth_dataset = NsynthDataset("datasets/nsynth")

        dat = nsynth_dataset[0]
        
        print(type(dat['pitch']), dat['pitch'].shape)
        print(type(dat['audio']), dat['audio'].shape)
        



        

    else:
        print("Choose an implemented dataset")
    