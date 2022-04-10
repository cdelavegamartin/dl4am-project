from distutils.debug import DEBUG
from unittest import skip
import torch
import torch.nn.functional as F
import resampy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
from time import gmtime, strftime
import librosa
from src.model import PitchExtractor, InstrumentClassifier
from utils.data import MdbStemSynthDataset, NsynthDataset
from utils.data import preprocess_data_for_crepe, create_bins, pitch_to_activation






def train_crepe(train_dataset='mdb'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")


    mdb_dataset = MdbStemSynthDataset("datasets/MDB-stem-synth/audio_stems","datasets/MDB-stem-synth/annotation_stems")

    model_sr = 16000
    p_ext = PitchExtractor(model_sr,model_size='medium')
    p_ext.load_state_dict(torch.load("trained_models/model_w_loss=0.008820420191903546.pth"))

    bins = create_bins(f_min=32.7, f_max=1975.5, n_bins=360)
    bins = bins.to(device)


    opt = torch.optim.Adam(p_ext.parameters(), lr=1e-4)
    loss = torch.nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

    print(type(scheduler.get_last_lr()), scheduler.get_last_lr())

    # writer = SummaryWriter(os.path.join("runs", f"run-{int(time.time()) }"), flush_secs=30)
    writer = SummaryWriter(os.path.join("runs", f"run-1649356020"), purge_step=40000, flush_secs=30)

    epochs =1
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

        frames,predictions = preprocess_data_for_crepe(stem)
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
                frames,predictions = preprocess_data_for_crepe(stem)
                
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

def train_instrument_classifier(train_dataset='nsynth'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")

    print("Loading training dataset")
    nsynth_dataset_train = NsynthDataset("datasets/nsynth",split='train')
    print("Loading validation dataset")
    nsynth_dataset_valid = NsynthDataset("datasets/nsynth", split='valid')
    print("Loading test dataset")
    nsynth_dataset_test = NsynthDataset("datasets/nsynth", split='test')

    

    model_sr = 16000
    model_size='large-shallow'
    n_mfcc=20
    n_classes=11
    
    model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc,
                                 input_length=126, n_classes=n_classes, model_size=model_size)

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

    # print(type(scheduler.get_last_lr()), scheduler.get_last_lr())

    # writer = SummaryWriter(os.path.join("runs", f"run-{int(time.time()) }"), flush_secs=30)
    writer = SummaryWriter(os.path.join("runs", "instclass", model_size, strftime("%Y-%m-%d_%H-%M-%S", gmtime())), purge_step=0, flush_secs=30)

    epochs =200
    batch_size = 500

    if len(nsynth_dataset_valid)%batch_size:
        n_batch_valid= len(nsynth_dataset_valid)//batch_size+1
    else:
        n_batch_valid= len(nsynth_dataset_valid)//batch_size

    if len(nsynth_dataset_test)%batch_size:
        n_batch_test= len(nsynth_dataset_test)//batch_size+1
    else:
        n_batch_test= len(nsynth_dataset_test)//batch_size

    

    
    step = 0
    
    val_counter = 0

    min_loss_valid = np.inf

    multi_thread = True
    if multi_thread:
        num_workers=4
        pin_mem = True
    else:
        num_workers=0
        pin_mem = False

    print("Loading training dataset")
    dataloader_train = DataLoader(nsynth_dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    
    dataloader_valid = DataLoader(nsynth_dataset_valid, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    
    dataloader_test = DataLoader(nsynth_dataset_test, batch_size=len(nsynth_dataset_test),
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)    

    for epoch in tqdm(range(epochs)):



        for batch in dataloader_train:
            # print(batch)
            
            mfcc = batch['mfcc'].to(device)
            target = batch['instrument'].to(device)
            prediction = model(mfcc)
            loss_train = loss(prediction,target)
            writer.add_scalar("loss_train", loss_train.item(), step)
            step += 1
            # print(f"Loss: {output}")
            opt.zero_grad()
            loss_train.backward()
            opt.step()
            # print('todebug')
            


        
        
        writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
        scheduler.step()
        
        # validation
        if not epoch % 5:
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
                    torch.save(model.state_dict(), f'trained_models/instclass/{model_size}/model.pth')
                    val_counter=0
                else:
                    val_counter += 1

        if val_counter>5:
            break

    # evaluate model
    torch.cuda.empty_cache() 
    evaluate=False
    if evaluate:
        # model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc,
        #                          input_length=126, n_classes=n_classes, model_size=model_size)
        model.load_state_dict(torch.load(f'trained_models/instclass/{model_size}/model.pth'))
        loss_test = 0
        with torch.no_grad():
                    for batch in dataloader_test:
                        mfcc = batch['mfcc'].to(device)
                        target = batch['instrument'].to(device)
                        prediction = model(mfcc)
                        loss_test += loss(prediction,target)

                    loss_test /=n_batch_test
                    writer.add_scalar("loss_test", loss_test.item(), step)
    
def train_combined_model(train_dataset='nsynth'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")

    # print("Loading training dataset")
    # nsynth_dataset_train = NsynthDataset("datasets/nsynth",split='train')
    # print("Loading validation dataset")
    # nsynth_dataset_valid = NsynthDataset("datasets/nsynth", split='valid')
    print("Loading test dataset")
    nsynth_dataset_test = NsynthDataset("datasets/nsynth", split='test')

    

    model_sr = 16000
    model_size='large-shallow'
    n_mfcc=20
    n_classes=11
    
    # model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc,
    #                              input_length=126, n_classes=n_classes, model_size=model_size)

    # model.to(device)

    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss = torch.nn.BCEWithLogitsLoss()

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99, last_epoch=-1, verbose=True)

    writer = SummaryWriter(os.path.join("runs", "instclass", model_size, strftime("%Y-%m-%d_%H-%M-%S", gmtime())), purge_step=0, flush_secs=30)

    epochs =200
    batch_size = 500

    # if len(nsynth_dataset_valid)%batch_size:
    #     n_batch_valid= len(nsynth_dataset_valid)//batch_size+1
    # else:
    #     n_batch_valid= len(nsynth_dataset_valid)//batch_size

    if len(nsynth_dataset_test)%batch_size:
        n_batch_test= len(nsynth_dataset_test)//batch_size+1
    else:
        n_batch_test= len(nsynth_dataset_test)//batch_size

    

    
    step = 0
    
    val_counter = 0

    min_loss_valid = np.inf

    multi_thread = True
    if multi_thread:
        num_workers=4
        pin_mem = True
    else:
        num_workers=0
        pin_mem = False

    # print("Loading training dataset")
    # dataloader_train = DataLoader(nsynth_dataset_train, batch_size=batch_size,
    #                     shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    
    # dataloader_valid = DataLoader(nsynth_dataset_valid, batch_size=batch_size,
    #                     shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    
    dataloader_test = DataLoader(nsynth_dataset_test, batch_size=len(nsynth_dataset_test),
                        shuffle=True, num_workers=num_workers, pin_memory=pin_mem)  


    for batch in dataloader_test:
            print(batch)
            break

    

    # for epoch in tqdm(range(epochs)):



    #     for batch in dataloader_train:
    #         # print(batch)
            
    #         mfcc = batch['mfcc'].to(device)
    #         target = batch['instrument'].to(device)
    #         prediction = model(mfcc)
    #         loss_train = loss(prediction,target)
    #         writer.add_scalar("loss_train", loss_train.item(), step)
    #         step += 1
    #         # print(f"Loss: {output}")
    #         opt.zero_grad()
    #         loss_train.backward()
    #         opt.step()
    #         # print('todebug')
            


        
        
    #     writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
    #     scheduler.step()
        
    #     # validation
    #     if not epoch % 5:
    #         loss_valid = 0
    #         with torch.no_grad():
    #             for batch in dataloader_valid:
    #                 mfcc = batch['mfcc'].to(device)
    #                 target = batch['instrument'].to(device)
    #                 prediction = model(mfcc)
    #                 loss_valid += loss(prediction,target)

                
    #             loss_valid /=n_batch_valid
    #             writer.add_scalar("loss_valid", loss_valid.item(), step)

    #             if min_loss_valid > loss_valid:
    #                 print(f'Validation Loss Decreased({min_loss_valid:.6f}--->{loss_valid:.6f}) \t Saving The Model')
    #                 min_loss_valid = loss_valid
    #                 # Saving State Dict
    #                 torch.save(model.state_dict(), f'trained_models/instclass/{model_size}/model.pth')
    #                 val_counter=0
    #             else:
    #                 val_counter += 1

    #     if val_counter>5:
    #         break

    # # evaluate model
    # torch.cuda.empty_cache() 
    # evaluate=False
    # if evaluate:
    #     # model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc,
    #     #                          input_length=126, n_classes=n_classes, model_size=model_size)
    #     model.load_state_dict(torch.load(f'trained_models/instclass/{model_size}/model.pth'))
    #     loss_test = 0
    #     with torch.no_grad():
    #                 for batch in dataloader_test:
    #                     mfcc = batch['mfcc'].to(device)
    #                     target = batch['instrument'].to(device)
    #                     prediction = model(mfcc)
    #                     loss_test += loss(prediction,target)

    #                 loss_test /=n_batch_test
    #                 writer.add_scalar("loss_test", loss_test.item(), step)
    

if __name__ == "__main__":
    # train_crepe()
    # train_instrument_classifier(train_dataset='nsynth')
    train_combined_model()

