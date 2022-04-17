from __future__ import print_function, division
import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import librosa

def preprocess_sample_for_crepe(audio, pitch, samplerate=16000, hop=0.01, window_size=1024, sample_start_s=0, n_frames=200):
    
    sample_start_samples = int(sample_start_s*samplerate)
    hop_length_samples = int(hop*samplerate)
    time_annot = pitch[:,0]
    pitch_annot  = pitch[:,1]
    time_annot_samples = (time_annot*samplerate).astype(int)
    frame_list = []
    prediction_list = []
    time_list=[]

    # pad the audio
    audio = np.pad(audio,(window_size//2, window_size//2))
    # print(audio.shape)
    start = sample_start_samples
    while len(prediction_list) < n_frames:
        end = start+window_size
        if end > len(audio):
            start=0
            continue
        f0 = pitch_annot[np.argmin(np.abs(time_annot_samples-start))]
        if f0 <10:
            start += hop_length_samples
            continue

        frame_list.append(np.expand_dims(audio[start:end],0))
        prediction_list.append(f0)
        time_list.append(start/samplerate)
        start += hop_length_samples

    frames = torch.from_numpy(np.vstack(frame_list))
    # print(f"frames.shape: {frames.shape}")
    pitches = torch.from_numpy(np.array(prediction_list, dtype=np.float32))
    timestamps = torch.from_numpy(np.array(time_list, dtype=np.float32))
    # print(f"pitches.shape: {pitches.shape}")
    frames_mean = torch.mean(frames,dim=1,keepdim=True)
    # print(f"frames_mean.shape: {frames_mean.shape}")
    frames_std = torch.std(frames,dim=1,keepdim=True)
    # print(f"frames_std.shape: {frames_std.shape}")
    frames = (frames-frames_mean)/frames_std
    # print(f"frames norm max: {frames.max()}, frames norm min: {frames.min()}")
    # print(f"frames norm mean: {frames.max()}, frames norm min: {frames.min()}")
    

    return frames, pitches, timestamps    

def freq_to_cents(frequency, f_ref=10.0):
    cents = 1200*torch.log2(frequency/f_ref)

    return cents
def cents_to_freq(cents, f_ref=10.0):
    frequency = f_ref*2**(cents/1200)

    return frequency

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
    # print("In pitch to act, cents min: ", torch.min(cents), "cents max: ", torch.max(cents))
    
    # print(f"cents: {cents}")
    # activations should be shape (total_frames,n_bins)
    bins = bins.reshape(1,-1)
    cents = cents.reshape(-1,1)
    activation = torch.exp(-(bins.expand(cents.shape[0],bins.shape[1])-cents.expand(cents.shape[0],bins.shape[1]))**2/(2*25**2))
    # print("In pitch to act, activation min: ", torch.min(activation), "activation max: ", torch.max(activation))
    return activation

def activation_to_pitch(activation, bins, n_bins_win=10):
    """activations should be shape (total_frames,n_bins)"""
    bin_max_i = torch.argmax(activation, dim=1, keepdim=True)
    confidence = bin_max_i
    
    
    bins = bins.reshape(1,-1)
    cents = torch.sum(activation*bins.expand(activation.shape[0],bins.shape[1]), dim=1)/torch.sum(activation, dim=1) 
    # frequencies = cents_to_freq(cents)
    
    return cents


class MdbStemSynthDataset(Dataset):
    def __init__(self, root_dir, sr=None, sample_start_s=None, n_frames=200, hop_s=0.01, transform=None):

        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "audio_stems")
        self.annot_dir = os.path.join(root_dir, "annotation_stems")
        self.sr = sr
        
        self.n_frames = n_frames
        self.sample_start_s = sample_start_s
        
        self.hop_s =hop_s
        self.bins = create_bins(f_min=32.7, f_max=1975.5, n_bins=360)
        
        wav_count = sum(1 for file in os.listdir(self.wav_dir) if (file.endswith(".wav") and not file.startswith("._")))
        annot_count = sum(1 for file in os.listdir(self.annot_dir) if file.endswith(".csv"))

        if wav_count == annot_count:
            self.data_fnames = [os.path.splitext(file)[0]  for file in os.listdir(self.annot_dir) if file.endswith(".csv")]
        else:
            print("number of audio files and annotations do not match")


    def __len__(self):
        return len(self.data_fnames)


    def __getitem__(self,index):
        
        name = self.data_fnames[index]
        audio, samplerate = librosa.load(os.path.join(self.wav_dir, name+".wav"),sr=self.sr, dtype=np.float32)
        # audio, samplerate = torchaudio.load(os.path.join(self.wav_dir, name+".wav"))
        # if samplerate != self.sr:
        #     audio = torchaudio.functional.resample(audio, samplerate, self.sr)
        #     samplerate = self.sr
            
        pitch_annotation= pd.read_csv(os.path.join(self.annot_dir, name)+".csv",header=None).to_numpy()
        # print("In Dataset get, Pitch annotation contains Nan: ", np.isnan(pitch_annotation).any())
        
        # if sample_start_s is not fixed, each sample starts at a random location in the audio stem
        if self.sample_start_s is None:
            start_s = np.random.rand()*len(audio)/samplerate
        else:
            start_s = self.sample_start_s
        
        frames, pitches, timestamps = preprocess_sample_for_crepe(audio, pitch_annotation, 
                                                      samplerate=samplerate, hop=self.hop_s, 
                                                      window_size=1024, 
                                                      sample_start_s=start_s, 
                                                      n_frames=self.n_frames)

        # print("In Dataset get, pitches contains Nan: ", torch.isnan(pitches).any())
        # print("In Dataset get, pitches min: ", torch.min(pitches), "pitches max: ", torch.max(pitches))
        # print(wave.open(os.path.join(self.wav_dir, name)+".wav").getframerate())
        
        pitch_one_hot = pitch_to_activation(pitches, self.bins)

        sample = {'name': name, 'samplerate': samplerate, 'frames':frames, 'pitch': pitch_one_hot, 'time':timestamps}
        
        return sample


class NsynthDataset(Dataset):
    def __init__(self, root_dir, split='test', sr=None, pitch_notation='hz', comb_mode=True):

        self.root_dir = root_dir
        self.split = split
        self.sr = sr
        self.pitch_notation = pitch_notation
        self.mode = comb_mode
        self.split_dir = os.path.join(self.root_dir,f"nsynth-{split}")
        self.json_data = pd.read_json(os.path.join(self.split_dir,"examples.json")).T

        self.n_fft = 2048
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
        

        audio, samplerate = torchaudio.load(os.path.join(self.split_dir, "audio", name+".wav"))
        audio = audio.squeeze(0)
        
        instrument = sample_data.instrument_family
        instrument_one_hot = torch.zeros(len(self.instrument_id_map))
        instrument_one_hot[instrument] = 1
        
        
        
        if self.mode:
            pitch_midi = sample_data.pitch
            
            if self.pitch_notation=='hz':
                pitch_hz = 2**((pitch_midi-69)/12)*440
                # pitch_annotation = torch.from_numpy(np.asarray([0.0,pitch_hz]).reshape(1,-1))
            else:
                print("pitch notation is not set or invalid for the dataset")
            specgram_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
            specgram = specgram_transform(audio)[1:,:]        
            n_frames= len(audio)//self.hop_length+1
            pitch_one_hot = pitch_to_activation(torch.tensor(n_frames * [(pitch_hz,)]), self.bins)
            
            return {'name': name, 'samplerate': samplerate, 'specgram':specgram, 
                    'pitch': pitch_one_hot, 'instrument': instrument_one_hot}
        
        else:
            mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=20, log_mels=True,
                                                    melkwargs={'hop_length':self.hop_length, 'n_fft':self.n_fft} )
            mfcc = torch.t(mfcc_transform(audio))

            return {'name': name, 'samplerate': samplerate, 'mfcc':mfcc, 'instrument': instrument_one_hot}



if __name__ == "__main__":

    dataset='mdb'

    if dataset=='mdb':

        mdb_dataset = MdbStemSynthDataset(root_dir="/import/c4dm-datasets/MDB-stem-synth/",
                                          sr=16000, sample_start_s=50, n_frames=2, hop_s=0.01)
        
        dat = mdb_dataset[0]

        print(type(dat['pitch']), dat['pitch'].shape)
        print(type(dat['frames']), dat['frames'].shape)
        

    elif dataset=='nsynth':

        nsynth_dataset = NsynthDataset("datasets/nsynth")

        dat = nsynth_dataset[0]
        
        print("inst", type(dat['instrument']), dat['instrument'].shape, dat['instrument'])
        print("mfcc", type(dat['mfcc']), dat['mfcc'].shape)
        print("pitch", type(dat['pitch']), dat['pitch'].shape)
        print("specgram", type(dat['specgram']), dat['specgram'].shape)
        print(len(nsynth_dataset.instrument_id_map))
        



        

    else:
        print("Choose an implemented dataset")
    