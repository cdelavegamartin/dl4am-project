from distutils.debug import DEBUG
from unittest import skip
import torch
import torch.nn.functional as F
import resampy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import os
import argparse
import pathlib
from time import time, gmtime, strftime
import librosa
from src.model import CombinedModel, PitchExtractor, InstrumentClassifier
from utils.data import MdbStemSynthDataset, NsynthDataset



def load_nsynth_dataset(dataset_dir):
    
    dataset_train = NsynthDataset(dataset_dir,split='train')
    dataset_valid = NsynthDataset(dataset_dir, split='valid')
    dataset_test = NsynthDataset(dataset_dir, split='test')
    
    return dataset_train, dataset_valid, dataset_test

def load_mdb_dataset(dataset_dir, train_split=0.8, valid_split=0.1, seed=None):

    mdb_dataset = MdbStemSynthDataset(root_dir=dataset_dir, sr=16000, 
                                      sample_start_s=None, n_frames=1000, hop_s=0.01)
    
    # Split dataset
    train_length=int(train_split* len(mdb_dataset))
    valid_length=int(valid_split* len(mdb_dataset))
    test_length=len(mdb_dataset)-train_length-valid_length
    lengths = [train_length, valid_length, test_length]
    
    # manual seed is used to be able to replicate test subset of the dataset
    if seed is None:
        dataset_train, dataset_valid, dataset_test = random_split(mdb_dataset, lengths)
    else:
        dataset_train, dataset_valid, dataset_test = random_split(mdb_dataset, lengths, 
                                                                  generator=torch.Generator().manual_seed(seed))
        
    return dataset_train, dataset_valid, dataset_test

def create_dataloaders(datasets, batch_size, multi_thread=True, num_workers=10):
    
    dataset_train, dataset_valid, dataset_test = datasets

    multi_thread = True
    if multi_thread:
        num_workers=10
        pin_mem = True
    else:
        num_workers=0
        pin_mem = False

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    
    return dataloader_train, dataloader_valid, dataloader_test  



def train_pitch(dataset_dir="datasets/MDB-stem-synth/", model_name='pitchext', model_size='medium', dropout=False, gpu_id=""):


    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")

    if dropout:
        model_name=f"{model_name}-dropout"
    
    
    # Initialize model
    model_size=model_size
    n_bins=360
    model_sr = 16000
    model = PitchExtractor(model_sr, n_bins=n_bins, model_size=model_size, dropout=dropout)
    model.to(device)
    
    

    # Optimizer (with lr scheduler) and loss
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

    
    # Directories for the saved models and tensorboard logging
    writer_dir = pathlib.Path(pathlib.Path(__file__).parent, "runs", model_name, model_size)
    writer_dir.mkdir(parents=True, exist_ok=True) 
    trained_models_dir = pathlib.Path(pathlib.Path(__file__).parent, "trained_models", model_name, model_size)
    trained_models_dir.mkdir(parents=True, exist_ok=True) 
    time_id = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(pathlib.Path(writer_dir, time_id), purge_step=0, flush_secs=30)
    
    # Training parameters
    epochs =200
    batch_size = 5
    rn_seed =38
    datasets = load_mdb_dataset(dataset_dir, train_split=0.8, valid_split=0.1, seed=rn_seed)
    dataloader_train, dataloader_valid, dataloader_test = create_dataloaders(datasets, 
                                                                             batch_size, 
                                                                             multi_thread=True, num_workers=10)
    
    # Save random seed to tensorboard for reuse in eval
    writer.add_scalar("Seed", rn_seed, 0)
    
    # Used for averaging validation loss
    if len(datasets[1])%batch_size:
        n_batch_valid= len(datasets[1])//batch_size+1
    else:
        n_batch_valid= len(datasets)//batch_size

    # Max number of batches per epoch during training
    n_batch = n_batch_valid
   
    
    testing=False
    if testing:
        
         # debug batch
        print('ready to batch')
        # tinit = time()
        for batch in dataloader_test:
            # print(type(batch))
            # tinit2 = time()
            frames = batch['frames']
            frames = frames.reshape(-1,frames.shape[-1]).to(device)
            target = batch['pitch']
            target = target.reshape(-1,target.shape[-1]).to(device)
            prediction = model(frames)
            loss_train = loss(prediction,target)
            # print(f"Loss: {loss_train}")
            # tnow = time()
            # print("secs: ", tnow-tinit,"secs2: ", tnow-tinit2,)
        return
    
    # Training loop
    print('Training...')
    min_loss_valid = np.inf
    step = 0
    val_counter = 0
    for epoch in tqdm(range(epochs)):

        model.train()
        for i_batch, batch in enumerate(dataloader_train):
            
            frames = batch['frames']
            frames = frames.reshape(-1,frames.shape[-1]).to(device)
            target = batch['pitch']
            target = target.reshape(-1,target.shape[-1]).to(device)
            prediction = model(frames)
            loss_train = loss(prediction,target)
            
            # monitor GPU usage
            if i_batch == 0 and not epoch % 5:
                print("mem reserved (GB): ", 9.31e-10*torch.cuda.memory_reserved(0))
                print("mem allocated (GB): ", 9.31e-10*torch.cuda.memory_allocated(0))
            
            writer.add_scalar("loss_train", loss_train.item(), step)
            opt.zero_grad()
            loss_train.backward()
            opt.step()
            step += 1
                        
            
                
            # limit the number of batches we train on per epoch
            if i_batch == n_batch:
                break
            


        
        
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        scheduler.step()
        
        # validation
        if not epoch % 5:
            model.eval()
            loss_valid = 0
            with torch.no_grad():
                for batch in dataloader_valid:
                    frames = batch['frames']
                    frames = frames.reshape(-1,frames.shape[-1]).to(device)
                    target = batch['pitch']
                    target = target.reshape(-1,target.shape[-1]).to(device)
                    prediction = model(frames)
                    loss_valid += loss(prediction,target)

                
                loss_valid /=n_batch_valid
                writer.add_scalar("loss_valid", loss_valid.item(), step)

                if min_loss_valid > loss_valid:
                    print(f'Validation Loss Decreased({min_loss_valid:.6f}--->{loss_valid:.6f}) \t Saving The Model')
                    min_loss_valid = loss_valid
                    # Saving State Dict
                    torch.save(model.state_dict(), pathlib.Path(trained_models_dir, f"model-{time_id}.pth"))
                    val_counter=0
                else:
                    val_counter += 1

        if val_counter>10:
            break
    
    graph = False
    if graph: 
        model.train()
        writer.add_graph(model, datasets[2][0])


def train_instrument_classifier(dataset_dir="datasets/nsynth/", model_name='instclass', model_size='medium', dropout=False, gpu_id=""):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    if dropout:
        model_name=f"{model_name}-dropout"
    
    # Initialize model
    model_sr = 16000
    model_size=model_size
    n_mfcc=20
    n_classes=11
    model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc, input_length=126, 
                                 n_classes=n_classes, model_size=model_size, dropout=dropout)

    model.to(device)
    
    # Optimizer (with lr scheduler) and loss
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

    # Directories for the saved models and tensorboard logging
    writer_dir = pathlib.Path(pathlib.Path(__file__).parent, "runs", model_name, model_size)
    writer_dir.mkdir(parents=True, exist_ok=True) 
    trained_models_dir = pathlib.Path(pathlib.Path(__file__).parent, "trained_models", model_name, model_size)
    trained_models_dir.mkdir(parents=True, exist_ok=True) 
    time_id = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(pathlib.Path(writer_dir, time_id), purge_step=0, flush_secs=30)



    # Training parameters
    epochs = 300
    batch_size = 6000
    datasets = load_nsynth_dataset(dataset_dir)
    dataloader_train, dataloader_valid, dataloader_test = create_dataloaders(datasets, 
                                                                             batch_size, 
                                                                             multi_thread=True, num_workers=10)
    

    # Used for averaging validation loss
    if len(datasets[1])%batch_size:
        n_batch_valid= len(datasets[1])//batch_size+1
    else:
        n_batch_valid= len(datasets)//batch_size

    # Max number of batches per epoch during training
    n_batch = n_batch_valid



    # Training loop
    print('Training...')
    min_loss_valid = np.inf
    step = 0
    val_counter = 0
    for epoch in tqdm(range(epochs)):

        model.train()

        for i_batch, batch in enumerate(dataloader_train):

            mfcc = batch['mfcc'].to(device)
            target = batch['instrument'].to(device)
            prediction = model(mfcc)
            loss_train = loss(prediction,target)
            writer.add_scalar("loss_train", loss_train.item(), step)
            
            # monitor GPU usage
            if i_batch == 0 and not epoch % 5:
                print("mem reserved (GB): ", 9.31e-10*torch.cuda.memory_reserved(0))
                print("mem allocated (GB): ", 9.31e-10*torch.cuda.memory_allocated(0))
            
            opt.zero_grad()
            loss_train.backward()
            opt.step()
            step += 1
        
        
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        scheduler.step()
        
        # validation
        if not epoch % 5:
            model.eval()
            loss_valid = 0
            with torch.no_grad():
                for batch in dataloader_valid:
                    mfcc = batch['mfcc'].to(device)
                    target = batch['instrument'].to(device)
                    prediction = model(mfcc)
                    loss_valid += loss(prediction,target)

                
                loss_valid /=n_batch_valid
                writer.add_scalar("loss_valid", loss_valid.item(), step)

                if min_loss_valid > loss_valid:
                    print(f'Validation Loss Decreased({min_loss_valid:.6f}--->{loss_valid:.6f}) \t Saving The Model')
                    min_loss_valid = loss_valid
                    # Saving State Dict
                    torch.save(model.state_dict(), pathlib.Path(trained_models_dir, f"model-{time_id}.pth"))
                    
                    val_counter=0
                else:
                    val_counter += 1

        if val_counter>10:
            break

    graph = False
    if graph: 
        model.train()
        writer.add_graph(model, datasets[2][0])
    
    
def train_combined_model(dataset_dir='datasets/nsynth', model_name='combined', model_size='small', dropout=False, gpu_id=""):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    if dropout:
        model_name=f"{model_name}-dropout"

    # Initialize model
    model_sr = 16000
    model_size=model_size
    n_bins=360
    n_classes=11
    input_length = 126
    model = CombinedModel(samplerate=model_sr, n_classes=n_classes, 
                          n_bins=n_bins, input_length=input_length, 
                          model_size=model_size)

    model.to(device)

    # Optimizer (with lr scheduler) and loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

    # Directories for the saved models and tensorboard logging
    writer_dir = pathlib.Path(pathlib.Path(__file__).parent, "runs", model_name, model_size)
    writer_dir.mkdir(parents=True, exist_ok=True) 
    trained_models_dir = pathlib.Path(pathlib.Path(__file__).parent, "trained_models", model_name, model_size)
    trained_models_dir.mkdir(parents=True, exist_ok=True) 
    time_id = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(pathlib.Path(writer_dir, time_id), purge_step=0, flush_secs=30)

    # Training parameters
    epochs = 200
    batch_size = 40
    datasets = load_nsynth_dataset(dataset_dir)
    dataloader_train, dataloader_valid, dataloader_test = create_dataloaders(datasets, 
                                                                             batch_size, 
                                                                             multi_thread=True, num_workers=10)
    

    # Used for averaging validation loss
    if len(datasets[1])%batch_size:
        n_batch_valid= len(datasets[1])//batch_size+1
    else:
        n_batch_valid= len(datasets)//batch_size

    # Max number of batches per epoch during training
    n_batch = n_batch_valid

   
    # Training loop
    print('Training...')
    min_loss_valid = np.inf
    step = 0
    val_counter = 0
    loss_weight = 5.0  # empirical pre-factor for the losses to have similar scale
    for epoch in tqdm(range(epochs)):
        
        model.train()
        for i_batch, batch in enumerate(dataloader_train):
            # print(batch)
            
            specgram = batch['specgram'].to(device)
            target_pitch = batch['pitch'].to(device)
            target_instrument = batch['instrument'].to(device)
            pred_pitch, pred_instrument = model(specgram)
            loss_train_pitch = loss(pred_pitch, target_pitch)
            loss_train_instr = loss(pred_instrument, target_instrument)
            loss_train = loss_weight*loss_train_pitch + loss_train_instr
            writer.add_scalar("loss_train_pitch", loss_train_pitch.item(), step)
            writer.add_scalar("loss_train_instr", loss_train_instr.item(), step)
            writer.add_scalar("loss_train", loss_train.item(), step)
            step += 1
            opt.zero_grad()
            loss_train.backward()
            opt.step()

            # limit the number of batches we train on per epoch
            if i_batch == n_batch:
                break
 
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        scheduler.step()
        
        # validation
        if not epoch % 5:
            loss_valid = 0
            loss_valid_pitch = 0
            loss_valid_instr = 0
            model.eval()
            with torch.no_grad():
                for batch in dataloader_valid:
                    specgram = batch['specgram'].to(device)
                    target_pitch = batch['pitch'].to(device)
                    target_instrument = batch['instrument'].to(device)
                    pred_pitch, pred_instrument = model(specgram)
                    loss_valid_pitch += loss(pred_pitch, target_pitch)
                    loss_valid_instr += loss(pred_instrument, target_instrument)
                    


                loss_valid_pitch /=n_batch
                loss_valid_instr /=n_batch
                loss_valid = loss_weight*loss_valid_pitch + loss_valid_instr
                writer.add_scalar("loss_valid_pitch", loss_valid_pitch.item(), step)
                writer.add_scalar("loss_valid_instr", loss_valid_instr.item(), step)                
                writer.add_scalar("loss_valid", loss_valid.item(), step)

                if min_loss_valid > loss_valid:
                    print(f'Validation Loss Decreased({min_loss_valid:.6f}--->{loss_valid:.6f}) \t Saving The Model')
                    min_loss_valid = loss_valid
                    # Saving State Dict
                    torch.save(model.state_dict(), pathlib.Path(trained_models_dir, f"model-{time_id}.pth"))
                    val_counter=0
                else:
                    val_counter += 1

        if val_counter>10:
            break
        
    graph = False
    if graph: 
        model.train()
        writer.add_graph(model, datasets[2][0])

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train the models')
    parser.add_argument('--model', '-m', choices=['pitch','instr','comb'] , 
                        required=True, help='Model to be trained (pitch/instr/comb')
    parser.add_argument('--dataset', '-d', action='store', type=pathlib.Path, 
                        required=True, help='Location of the dataset')
    parser.add_argument('--regularization', '-r', action='store_true' , help='Use dropout regularization')
    parser.add_argument('--model_size', '-s', action='store' , required=True, help='Model size to use')
    parser.add_argument('--gpu_id', '-g', action='store', default="", help='GPU device to use')
    
    
    args = parser.parse_args()
    model = args.model
    dataset_dir=args.dataset
    dropout = args.regularization
    model_size = args.model_size
    gpu_id = args.gpu_id
    
    if model=='pitch':
        train_pitch(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    elif model=='instr':
        train_instrument_classifier(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    elif model=='comb':
        train_combined_model(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
        

