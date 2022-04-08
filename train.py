from unittest import skip
import torch
import torch.nn.functional as F
import resampy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import time


from src.model import PitchExtractor
from utils.data import MdBStemSynthDataset



def convert_data(stem, hop=0.01):
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


def map_to_activation(frequency, f_min=32.7, f_max=1975.5, n_bins=360):
    cents_min = freq_to_cents(torch.tensor(f_min))
    # print(f"cents_min: {cents_min}")
    cents_max = freq_to_cents(torch.tensor(f_max))
    # print(f"cents_max: {cents_max}")
    bins = torch.from_numpy(np.linspace(cents_min,cents_max,n_bins,dtype=np.float32))
    cents = freq_to_cents(frequency)
    # print(f"cents: {cents}")
    

    activation = np.exp(-(bins-cents)**2/(2*25**2))
    return activation


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



device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


mdb_dataset = MdBStemSynthDataset("datasets/MDB-stem-synth/audio_stems","datasets/MDB-stem-synth/annotation_stems")

model_sr = 16000
p_ext = PitchExtractor(model_sr,model_size='medium')
p_ext.load_state_dict(torch.load("trained_models/model_w_loss=0.008820420191903546.pth"))

bins = create_bins(f_min=32.7, f_max=1975.5, n_bins=360)
bins = bins.to(device)
# stem = mdb_dataset[0]
# print(f"bins.device={bins.device}")
# print(stem['samplerate'])
# audio_resampled = resampy.resample(stem['audio'], stem['samplerate'], model_sr)
# print(audio_resampled.shape)
# time_annot = stem['pitch'][:,0]
# print(time_annot.shape)

# frames,predictions = convert_data(stem)
# print('frames: ', frames)

opt = torch.optim.Adam(p_ext.parameters(), lr=1e-4)
loss = torch.nn.BCEWithLogitsLoss()

scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

print(type(scheduler.get_last_lr()), scheduler.get_last_lr())

# writer = SummaryWriter(os.path.join("runs", f"run-{int(time.time()) }"), flush_secs=30)
writer = SummaryWriter(os.path.join("runs", f"run-1649356020"), purge_step=40000, flush_secs=30)

epochs =10000
batch_size = 150
n_batch_per_epoch = 100
step = 40000
stem_n = 0
val_counter = 0

min_valid_loss = np.inf

for epoch in tqdm(range(epochs)):


    stem = mdb_dataset[np.random.randint(0,200)]
    stem['audio'] = resampy.resample(stem['audio'], stem['samplerate'], model_sr)
    stem['samplerate']=model_sr
    print('Training with: ', stem['name'])
    # time_annot = stem['pitch'][:,0]
    # print(time_annot.shape)

    frames,predictions = convert_data(stem)
    p_ext.to(device)
    frames = frames.to(device)
    frames = frames.unsqueeze(1)
    # print(f"frames.shape: {frames.shape}")
    predictions = predictions.to(device)
    y = pitch_to_activation(predictions, bins)


    for b in range(n_batch_per_epoch):
        inds = np.random.randint(0, predictions.shape[0],(batch_size,))
        bframes = frames[inds,:,:]
        by = y[inds,:]
        by_hat = p_ext(bframes)
        output = loss(by_hat,by)
        writer.add_scalar("loss", output.item(), step)
        step += 1
        # print(f"Loss: {output}")
        opt.zero_grad()
        output.backward()
        opt.step()

    
    
    writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
    scheduler.step()
    
    # validation
    if not epoch % 10:

        with torch.no_grad():
            stem = mdb_dataset[np.random.randint(200,230)]
            stem['audio'] = resampy.resample(stem['audio'], stem['samplerate'], model_sr)
            stem['samplerate']=model_sr
            print(f"validation with:", stem['name'])
            frames,predictions = convert_data(stem)
            
            frames = frames.to(device)
            frames = frames.unsqueeze(1)
            # print(f"frames.shape: {frames.shape}")
            predictions = predictions.to(device)
            y = pitch_to_activation(predictions, bins)


            inds = np.random.randint(0, predictions.shape[0],(batch_size,))
            bframes = frames[inds,:,:]
            by = y[inds,:]
            by_hat = p_ext(bframes)
            valid_loss = loss(by_hat,by)
            writer.add_scalar("val_loss", valid_loss.item(), step)

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(p_ext.state_dict(), f'trained_models/model_steps={step}.pth')
                val_counter=0
            else:
                val_counter += 1

    if val_counter>20:
        break