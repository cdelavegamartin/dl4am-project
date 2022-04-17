import os
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from torch.utils.data import DataLoader, random_split
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import crepe
import pathlib
import argparse
import matplotlib.pyplot as plt
from utils.data import MdbStemSynthDataset, NsynthDataset
from utils.data import activation_to_pitch, create_bins
from src.model import PitchExtractor, InstrumentClassifier, CombinedModel



def create_dataloader(dataset, batch_size, multi_thread=True, num_workers=10):
      

    if multi_thread:
        num_workers=5
        pin_mem = True
    else:
        num_workers=0
        pin_mem = False

    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    
    return dataloader


def plot_confusion_matrix(cm,
                      target_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True,
                      save_dir=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                see http://matplotlib.org/examples/color/colormaps_reference.html
                plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                If True, plot the proportions


    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    FONT_SIZE = 10

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    fig, ax = plt.subplots(figsize=(16, 14))    # 8, 6
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONT_SIZE)

    


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    fontsize=FONT_SIZE,
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    fontsize=FONT_SIZE,
                    color="white" if cm[i, j] > thresh else "black")


    # plt.tight_layout()
    ax.set_title(title,fontsize=22)
    ax.set_ylabel('True label',fontsize=16)
    ax.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=16)
    plt.show()
    if save_dir is not None:
        plt.savefig(pathlib.Path(save_dir,'conf_matrix.png'))
        

def load_mdb_dataset(dataset_dir, train_split=0.8, valid_split=0.1, seed=None):

    mdb_dataset = MdbStemSynthDataset(root_dir=dataset_dir, sr=16000, 
                                      sample_start_s=0, n_frames=1000, hop_s=0.01)
    
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

def evaluate_pitch(trained_model, dataset_dir='datasets/MDB-stem-synth/', gpu_id="", tols=[50,20,10]):
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    
    
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    eval_out_dir = pathlib.Path(pathlib.Path(__file__).parent, "eval", model_name, model_size, model_path.name.strip('.pth'))
    eval_out_dir.mkdir(parents=True, exist_ok=True) 
    # print(model_path.name.strip('.pth'))
    # print(eval_out_dir)
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False
    
    if not dataset_dir.exists():
        print (f"Dataset does not exist: {dataset_dir}")
        return
    
    # load test dataset
    _, _, dataset = load_mdb_dataset(dataset_dir, train_split=0.8, valid_split=0.1, seed=38)

    
    # Initialize model
    n_bins=360
    model_sr = 16000
    model = PitchExtractor(model_sr, n_bins=n_bins, model_size=model_size, dropout=dropout)

    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    
    # print(list(dataset.instrument_id_map.keys())[list(dataset.instrument_id_map.values()).index(0)])
    
    # Loss function
    loss = torch.nn.BCELoss()
    
    # Eval parameters
    # batch_size = len(dataset)
    batch_size = 5
    dataloader = create_dataloader(dataset, batch_size=batch_size, multi_thread=True, num_workers=4)
    
    # Used for averaging validation loss
    if len(dataset)%batch_size:
        n_batch= len(dataset)//batch_size+1
    else:
        n_batch= len(dataset)//batch_size
    
    n_batch = 5
    
    loss_test = 0
    pitch_true = []
    pitch_pred = []
    pitch_bins = create_bins(f_min=32.7, f_max=1975.5, n_bins=360).to(device)
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            # print ("batch num: ", i_batch)
            frames = batch['frames']
            frames = frames.reshape(-1,frames.shape[-1]).to(device)
            target = batch['pitch']
            # print("Target contains Nan: ", torch.isnan(target).any())
            target = target.reshape(-1,target.shape[-1]).to(device)
            prediction = model(frames)
            prediction = torch.sigmoid(prediction)
            pitch_true.extend(activation_to_pitch(target, pitch_bins).detach().tolist())
            # print("Pitch true contains Nan: ", np.isnan(np.array(pitch_true)).any())
            pitch_pred.extend(activation_to_pitch(prediction, pitch_bins).detach().tolist())
            # print("dtypes ", f"target.dtype={target.dtype}", f"prediction.dtype={prediction.dtype}")
            loss_test += loss(prediction,target)
            
            if i_batch==n_batch:
                break
            
            
        loss_test /=n_batch
        
    pitch_true = np.array(pitch_true)
    pitch_pred = np.array(pitch_pred)
    
    v_true = np.ones_like(pitch_true)
    v_pred = np.ones_like(pitch_pred)
    
    print("Length of pitch: ", len(pitch_true) )
    print("Max p true",np.max(pitch_true), "Max p pred",np.max(pitch_pred))
    print("Min p true",np.min(pitch_true), "Min p pred",np.min(pitch_pred))
    print("Pitch true contains Nan: ", np.isnan(pitch_true).any())
    print("Pitch pred contains Nan: ", np.isnan(pitch_pred).any())
    
    difference = pitch_true-pitch_pred
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.plot(range(len(pitch_true)), difference)
    ax.set_title('Predicted vs. true pitch',fontsize=22)
    ax.set_ylabel('Difference between y_true and y_pred (cents)',fontsize=16)
    ax.set_xlabel('Frames', fontsize=16)
    plt.show()
    plt.savefig(pathlib.Path(eval_out_dir,'cents_diff.png'))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.hist(difference, bins=np.arange(-3500,3500,50))
    ax.set_title('Predicted vs. true pitch Histogram',fontsize=22)
    ax.set_ylabel('Number of frames',fontsize=16)
    ax.set_xlabel('Difference between y_true and y_pred (cents)', fontsize=16)
    plt.show()
    plt.savefig(pathlib.Path(eval_out_dir,'cents_diff_hist.png'))
    plt.close()
    
    
    rpa = []
    rca = []
    for tol in tols:
        rpa.append(raw_pitch_accuracy(v_true, pitch_true, v_pred, pitch_pred, tol))
        rca.append(raw_chroma_accuracy(v_true, pitch_true, v_pred, pitch_pred, tol))
        
        
    with open(pathlib.Path(eval_out_dir, "results.dat"), "w", encoding = 'utf-8') as f:
        for i, tol in enumerate(tols):
            results = f"Tolerance: {tol}, RPA: {rpa[i]:.5f}, RCA: {rca[i]:.5f} \n"
            print(results)
            f.write(results)
        f.write(f"Loss: {loss_test:.5f} \n")
            
    
    return


def evaluate_instclass(trained_model, dataset_dir='datasets/nsynth/', gpu_id=""):
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    
    
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    eval_out_dir = pathlib.Path(pathlib.Path(__file__).parent, "eval", model_name, model_size, model_path.name.strip('.pth'))
    eval_out_dir.mkdir(parents=True, exist_ok=True) 
    # print(model_path.name.strip('.pth'))
    # print(eval_out_dir)
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False
    
    if not dataset_dir.exists():
        print (f"Dataset does not exist: {dataset_dir}")
        return
    
    # load test dataset
    dataset = NsynthDataset(dataset_dir, split='test', comb_mode=False)

    
    # Initialize model
    model_sr = 16000
    model_size=model_size
    n_mfcc=20
    n_classes=11
    model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc, input_length=126, 
                                 n_classes=n_classes, model_size=model_size, dropout=dropout)

    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    
    # print(list(dataset.instrument_id_map.keys())[list(dataset.instrument_id_map.values()).index(0)])
    
    # Loss function
    loss = torch.nn.BCELoss()
    
    # Eval parameters
    batch_size = len(dataset)
    # batch_size = 40
    dataloader = create_dataloader(dataset, batch_size=batch_size, multi_thread=True, num_workers=4)
    
    # Used for averaging validation loss
    if len(dataset)%batch_size:
        n_batch= len(dataset)//batch_size+1
    else:
        n_batch= len(dataset)//batch_size
    
    
    
    loss_test = 0
    labels_true = []
    labels_pred = []
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            mfcc = batch['mfcc'].to(device)
            target = batch['instrument'].to(device)
            prediction = model(mfcc)
            prediction = torch.sigmoid(prediction)
            # print(prediction.shape)
            labels_true.extend(torch.argmax(target,dim=1).detach().tolist())
            labels_pred.extend(torch.argmax(prediction,dim=1).detach().tolist())
            loss_test += loss(prediction,target)
            
            if i_batch>5:
                break
            
        loss_test /=n_batch
    # print(labels_pred, labels_true)
    # Purge
    for i, pclass in enumerate(labels_pred):
        if pclass==9:
            del labels_pred[i]
            del labels_true[i]
    
    print("set true:", set(labels_true), "set pred:", set(labels_pred))
    print("synth-lead true: ",sum(1 for i in labels_true if i == 9))
    print("synth-lead pred: ",sum(1 for i in labels_pred if i == 9))
    acc = accuracy_score(labels_true, labels_pred)
    print("Accuracy:",acc)
    display_labels = list(dataset.instrument_id_map.keys())
    display_labels.remove('synth_lead')
    
    
    cm = confusion_matrix(labels_true, labels_pred)
    # print("vocals right: ", cm[10,10])
    # fig = plt.figure(figsize=(50, 50))
    # ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, normalize='true', 
    #                                         cmap=plt.cm.Blues, 
    #                                         xticks_rotation=45)
    # # disp.plot(cmap=plt.cm.Blues, ax=ax)
    # plt.savefig(pathlib.Path(eval_out_dir,'conf_matrix.png'))
    
    plot_title = f"Instrument Classifier, size={model_size}, dropout={dropout}"
    plot_confusion_matrix(cm,
                      target_names = display_labels,
                      title=plot_title,
                      cmap=plt.cm.Blues,
                      normalize=True,
                      save_dir=eval_out_dir)
    
    with open(pathlib.Path(eval_out_dir, "results.dat"), "w", encoding = 'utf-8') as f:
        f.write(f"Loss: {loss_test:.5f} \n")
        f.write(f"Accuracy: {acc:.5f} \n")
        
    
    return


def evaluate_combined(trained_model, dataset_dir='datasets/nsynth/', gpu_id="", tols=[50,20,10]):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    
    
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    eval_out_dir = pathlib.Path(pathlib.Path(__file__).parent, "eval", model_name, model_size, model_path.name.strip('.pth'))
    eval_out_dir.mkdir(parents=True, exist_ok=True) 
    # print(model_path.name.strip('.pth'))
    # print(eval_out_dir)
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False
    
    if not dataset_dir.exists():
        print (f"Dataset does not exist: {dataset_dir}")
        return
    
    # load test dataset
    dataset = NsynthDataset(dataset_dir, split='test', comb_mode=True)

    
    # Initialize model
    model_sr = 16000
    n_bins=360
    n_classes=11
    input_length = 126
    model = CombinedModel(samplerate=model_sr, n_classes=n_classes, 
                          n_bins=n_bins, input_length=input_length, 
                          model_size=model_size)


    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    
    # print(list(dataset.instrument_id_map.keys())[list(dataset.instrument_id_map.values()).index(0)])
    
    # Loss function
    loss = torch.nn.BCELoss()
    
    # Eval parameters
    # batch_size = len(dataset)
    batch_size = 100
    dataloader = create_dataloader(dataset, batch_size=batch_size, multi_thread=True, num_workers=4)
    
    # Used for averaging validation loss
    if len(dataset)%batch_size:
        n_batch= len(dataset)//batch_size+1
    else:
        n_batch= len(dataset)//batch_size
    
    
    
    loss_test_pitch = 0
    loss_test_instr = 0
    pitch_true = []
    pitch_pred = []
    labels_true = []
    labels_pred = []
    pitch_bins = create_bins(f_min=32.7, f_max=1975.5, n_bins=360).to(device)
    loss_weight = 5.0  # empirical pre-factor for the losses to have similar scale
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            specgram = batch['specgram'].to(device)
            print("specgram shape: ",specgram.shape)
            target_pitch = batch['pitch'].to(device)
            # print("Target contains Nan: ", torch.isnan(target_pitch).any())
            target_instrument = batch['instrument'].to(device)
            pred_pitch, pred_instrument = model(specgram)
            pred_pitch = torch.sigmoid(pred_pitch)
            pred_instrument = torch.sigmoid(pred_instrument)
            # print("pitch shapes: ",target_pitch.shape, pred_pitch.shape)
            # print("Target contains Nan: ", torch.sum(torch.isnan(target_pitch)))
            pitch_true.extend(activation_to_pitch(target_pitch.reshape(-1,target_pitch.shape[-1]), 
                                                  pitch_bins).detach().tolist())
            pitch_pred.extend(activation_to_pitch(pred_pitch.reshape(-1,pred_pitch.shape[-1]), 
                                                  pitch_bins).detach().tolist())
            labels_true.extend(torch.argmax(target_instrument,dim=1).detach().tolist())
            labels_pred.extend(torch.argmax(pred_instrument,dim=1).detach().tolist())
            loss_test_pitch += loss(pred_pitch, target_pitch)
            loss_test_instr += loss(pred_instrument, target_instrument)

            
            
        loss_test_pitch /=n_batch
        loss_test_instr /=n_batch
        loss_test = loss_weight*loss_test_pitch + loss_test_instr
        
    pitch_true = np.array(pitch_true)
    pitch_pred = np.array(pitch_pred)
    
    pitch_pred = pitch_pred[~np.isnan(pitch_true)]
    pitch_true = pitch_true[~np.isnan(pitch_true)]
    
    v_true = np.ones_like(pitch_true)
    v_pred = np.ones_like(pitch_pred)
    # Purge
    for i, pclass in enumerate(labels_pred):
        if pclass==9:
            del labels_pred[i]
            del labels_true[i]
    
    rpa = []
    rca = []
    for tol in tols:
        rpa.append(raw_pitch_accuracy(v_true, pitch_true, v_pred, pitch_pred, tol))
        rca.append(raw_chroma_accuracy(v_true, pitch_true, v_pred, pitch_pred, tol))
    
    
    print("set true:", set(labels_true), "set pred:", set(labels_pred))
    print("synth-lead true: ",sum(1 for i in labels_true if i == 9))
    print("synth-lead pred: ",sum(1 for i in labels_pred if i == 9))
    print("vocal true: ",sum(1 for i in labels_true if i == 10))
    print("vocal pred: ",sum(1 for i in labels_pred if i == 10))
    
    difference = pitch_true-pitch_pred
    
    print("Length of pitch: ", len(pitch_true) )
    print("Max p true",np.max(pitch_true), "Max p pred",np.max(pitch_pred))
    print("Min p true",np.min(pitch_true), "Min p pred",np.min(pitch_pred))
    print("Pitch true contains Nan: ", np.isnan(pitch_true).any())
    print("Pitch pred contains Nan: ", np.isnan(pitch_pred).any())
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.plot(range(len(pitch_true)), difference)
    ax.set_title('Predicted vs. true pitch',fontsize=22)
    ax.set_ylabel('Difference between y_true and y_pred (cents)',fontsize=16)
    ax.set_xlabel('Frames', fontsize=16)
    plt.show()
    plt.savefig(pathlib.Path(eval_out_dir,'cents_diff.png'))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.hist(difference, bins=np.arange(-3500,3500,50))
    ax.set_title('Predicted vs. true pitch Histogram',fontsize=22)
    ax.set_ylabel('Number of frames',fontsize=16)
    ax.set_xlabel('Difference between y_true and y_pred (cents)', fontsize=16)
    plt.show()
    plt.savefig(pathlib.Path(eval_out_dir,'cents_diff_hist.png'))
    plt.close()
      
    acc = accuracy_score(labels_true, labels_pred)
    print("Accuracy:",acc)
    display_labels = list(dataset.instrument_id_map.keys())
    display_labels.remove('synth_lead')
    # print(display_labels)
    cm = confusion_matrix(labels_true, labels_pred)
    # print("vocals right: ", cm[10,10])
    # fig = plt.figure(figsize=(50, 50))
    # ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, normalize='true', 
    #                                         cmap=plt.cm.Blues, 
    #                                         xticks_rotation=45)
    # # disp.plot(cmap=plt.cm.Blues, ax=ax)
    # plt.savefig(pathlib.Path(eval_out_dir,'conf_matrix.png'))
    
    plot_title = f"Combined model, size={model_size}, dropout={dropout}"
    plot_confusion_matrix(cm,
                      target_names = display_labels,
                      title=plot_title,
                      cmap=plt.cm.Blues,
                      normalize=True,
                      save_dir=eval_out_dir)  
        
    with open(pathlib.Path(eval_out_dir, "results.dat"), "w", encoding = 'utf-8') as f:
        for i, tol in enumerate(tols):
            results = f"Tolerance: {tol}, RPA: {rpa[i]:.5f}, RCA: {rca[i]:.5f} \n"
            print(results)
            
            f.write(results)
        f.write(f"Loss: {loss_test:.5f}, Loss Pitch: {loss_test_pitch:.5f}, Loss Instr: {loss_test_instr:.5f} \n")
        f.write(f"Accuracy: {acc:.5f} \n")
            
            

    
    return









def evaluate_model(trained_model, dataset_dir, gpu_id=""):
    
    model_path = pathlib.Path.resolve(trained_model)
    model_name = model_path.parents[1].name
    if 'instclass' in model_name and 'nsynth' in dataset_dir.name:
        
        evaluate_instclass(trained_model=trained_model, dataset_dir=dataset_dir, gpu_id=gpu_id)
        
    elif 'combined' in model_name and 'nsynth' in dataset_dir.name:
        
        evaluate_combined(trained_model=trained_model, dataset_dir=dataset_dir, gpu_id=gpu_id, tols=[10,20,50,100,200, 600])
        
    elif 'pitch' in model_name and 'MDB-stem-synth' in dataset_dir.name:
        
        evaluate_pitch(trained_model=trained_model, dataset_dir=dataset_dir, gpu_id=gpu_id, tols=[10,20,50,100,200, 600])
    else:
        print('Not the right combination of model and dataset')
        
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model', '-m', action='store', type=pathlib.Path,
                        required=True, help='Model to be evaluated, or folder containing ')
    parser.add_argument('--dataset', '-d', action='store', type=pathlib.Path, 
                        required=True, help='Location of the datasets')
    parser.add_argument('--gpu_id', '-g', action='store', default="", help='GPU device to use')
    
    
    args = parser.parse_args()
    model = args.model
    dataset_dir=args.dataset
    gpu_id = args.gpu_id
    
    # if model=='pitch':
    #     train_pitch(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    # elif model=='instr':
    #     train_instrument_classifier(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    # elif model=='comb':
    #     train_combined_model(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    
    print(dataset_dir)
    if model.is_dir():
        for trained_model in model.rglob("*.pth"):
            print(trained_model)
            evaluate_model(trained_model=trained_model, dataset_dir=dataset_dir, gpu_id=gpu_id)
    elif model.is_file():
            evaluate_model(trained_model=model, dataset_dir=dataset_dir, gpu_id=gpu_id)
                
            
            
            
    # evaluate_model(trained_model)
        
        
    # evaluate_instclass(trained_model=model, dataset_dir=dataset_dir, gpu_id=gpu_id)
    
 