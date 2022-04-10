from __future__ import print_function, division
import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import librosa

def preprocess_data_for_crepe(stem, hop=0.01):
    window_size = 1024
    hop_length_samples = int(hop*stem['samplerate'])
    audio = stem['audio']
    time_annot = stem['pitch'][:,0]
    pitch_annot  = stem['pitch'][:,1]
    time_annot_samples = (time_annot*stem['samplerate']).astype(int)
    # print(f'audio shape is: {audio.shape}')
    total_frames = 1 + int(stem['audio'].shape[0]//hop_length_samples)
    # print(f'total_frames is: {total_frames}')
    frame_list = []
    prediction_list = []

    # pad the audio
    audio = np.pad(audio,(window_size//2, window_size//2))
    # print(audio.shape)
    for idf in range(total_frames):
        start  = idf*hop_length_samples
        f0 = pitch_annot[np.argmin(np.abs(time_annot_samples-start))]
        if f0 <10:
            continue

        frame_list.append(np.expand_dims(audio[start:start+window_size],0))
        prediction_list.append(pitch_annot[np.argmin(np.abs(time_annot_samples-start))])

    frames = torch.from_numpy(np.vstack(frame_list))
    # print(f"frames.shape: {frames.shape}")
    predictions = torch.from_numpy(np.array(prediction_list).reshape(-1,1))
    # print(f"predictions.shape: {predictions.shape}")
    frames_mean = torch.mean(frames,dim=1,keepdim=True)
    # print(f"frames_mean.shape: {frames_mean.shape}")
    frames_std = torch.std(frames,dim=1,keepdim=True)
    # print(f"frames_std.shape: {frames_std.shape}")
    frames = (frames-frames_mean)/frames_std
    # print(f"frames norm max: {frames.max()}, frames norm min: {frames.min()}")
    # print(f"frames norm mean: {frames.max()}, frames norm min: {frames.min()}")
    

    return frames, predictions    

def freq_to_cents(frequency):
    cents = 1200*torch.log2(frequency/10.0)

    return cents

def create_bins(f_min=32.7, f_max=1975.5, n_bins=360):
    cents_min = freq_to_cents(torch.tensor(f_min))
    # print(f"cents_min: {cents_min}")
    cents_max = freq_to_cents(torch.tensor(f_max))
    # print(f"cents_max: {cents_max}")
    bins = torch.from_numpy(np.linspace(cents_min,cents_max,n_bins,dtype=np.float32))
    return bins


def pitch_to_activation(frequencies, bins):
    """Frequencies should be shape (total_frames,1)"""
    cents = freq_to_cents(frequencies)
    # print(f"cents: {cents}")
    # activations should be shape (total_frames,n_bins)
    bins = bins.reshape(1,-1)
    cents = cents.reshape(-1,1)
    activation = torch.exp(-(bins.expand(cents.shape[0],bins.shape[1])-cents.expand(cents.shape[0],bins.shape[1]))**2/(2*25**2))
    return activation


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

        self.hop_length =512
        self.bins = create_bins(f_min=32.7, f_max=1975.5, n_bins=360)

        self.instrument_id_map = {'bass':0,
                                  'brass':1,
                            	  'flute':2,
                            	  'guitar':3,
                            	  'keyboard':4,
                            	  'mallet':5,
                            	  'organ':6,
                            	  'reed':7,
                            	  'string':8,
                            	  'synth_lead':9,
                             	  'vocal':10,
                                 }

    
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        
        sample_data = self.json_data.iloc[index]
        name = sample_data.note_str
        pitch_midi = sample_data.pitch
        instrument = sample_data.instrument_family
        if self.pitch_notation=='hz':
            pitch_hz = 2**((pitch_midi-69)/12)*440
            pitch_annotation = torch.from_numpy(np.asarray([0.0,pitch_hz]).reshape(1,-1))
        else:
            print("pitch notation is not set or invalid for the dataset")

        pitch_one_hot = pitch_to_activation(torch.tensor(pitch_hz), self.bins)

        # audio, samplerate = librosa.load(os.path.join(self.split_dir, "audio", name+".wav"),sr=self.sr)
        audio, samplerate = torchaudio.load(os.path.join(self.split_dir, "audio", name+".wav"))
        # TODO: This needs to be made into a transform that can be manipulated from outside
        mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=20, log_mels=True,
                                                    melkwargs={'hop_length':512, 'n_fft':2048} )

        specgram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512)
        audio = audio.squeeze(0)
        mfcc = torch.t(mfcc_transform(audio))
        specgram = specgram_transform(audio)[1:,:]

        
        n_frames= len(audio)//self.hop_length+1
        pitch_one_hot = pitch_to_activation(torch.tensor(n_frames * [(pitch_hz,)]), self.bins)
        

        instrument_one_hot = torch.zeros(len(self.instrument_id_map))
        instrument_one_hot[instrument] = 1
        instrument_one_hot = instrument_one_hot
        sample = {'name': name, 'audio': audio, 'samplerate': samplerate,
                  'pitch': pitch_one_hot, 'instrument': instrument_one_hot, 'mfcc':mfcc, 'specgram':specgram}

        return sample


class ToMFCC(object):
    """Convert ndarrays in sample to Tensors. Only audio"""

    def __call__(self, sample):

        sample_mod=sample
        print('Hey there, MFCC')
        
        return sample_mod


torchaudio.transforms

if __name__ == "__main__":

    dataset='nsynth'

    if dataset=='mdb':

        mdb_dataset = MdbStemSynthDataset("datasets/MDB-stem-synth/audio_stems","datasets/MDB-stem-synth/annotation_stems")
        
        dat = mdb_dataset[0]

        print(type(dat['pitch']), dat['pitch'].shape)
        print(type(dat['audio']), dat['audio'].shape)
        

    elif dataset=='nsynth':

        nsynth_dataset = NsynthDataset("datasets/nsynth",transform=ToMFCC())

        dat = nsynth_dataset[0]
        
        
        print("audio", type(dat['audio']), dat['audio'].shape)
        print("inst", type(dat['instrument']), dat['instrument'].shape, dat['instrument'])
        print("mfcc", type(dat['mfcc']), dat['mfcc'].shape)
        print("pitch", type(dat['pitch']), dat['pitch'].shape)
        print("specgram", type(dat['specgram']), dat['specgram'].shape)
        print(len(nsynth_dataset.instrument_id_map))
        



        

    else:
        print("Choose an implemented dataset")
    