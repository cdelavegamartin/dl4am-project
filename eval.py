import os
import torch
import numpy as np
import itertools
from torch.utils.data import DataLoader, random_split
import mir_eval
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import crepe
import pathlib
import argparse
import matplotlib.pyplot as plt
from utils.data import MdbStemSynthDataset, NsynthDataset
from src.model import PitchExtractor, InstrumentClassifier, CombinedModel


def extract_pitch_crepe():
    pass

def calculate_rpa(true_pitch, predicted_pitch):

    # true pitch and predicted pitch 
    pass

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


def plot_confusion_matrix_2(cm,
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
    FONT_SIZE = 8

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8*2, 6*2))    # 8, 6
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONT_SIZE)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


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


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    if save_dir is not None:
        plt.savefig(pathlib.Path(save_dir,'conf_matrix.png'))
        




def evaluate_class_accuracy(trained_model, dataset_dir='datasets/nsynth/', gpu_id=""):
    
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    
    
    
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print ("Model does not exist")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    eval_out_dir = pathlib.Path(pathlib.Path(__file__).parent, "eval", model_name, model_size, model_path.name.strip('.pth'))
    eval_out_dir.mkdir(parents=True, exist_ok=True) 
    print(model_path.name.strip('.pth'))
    print(eval_out_dir)
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False
        
    
    # load test dataset
    dataset = NsynthDataset(dataset_dir, split='valid', comb_mode=False)

    
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
    loss = torch.nn.BCEWithLogitsLoss()
    
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
            # print(prediction.shape)
            labels_true.extend(torch.argmax(target,dim=1).detach().tolist())
            labels_pred.extend(torch.argmax(prediction,dim=1).detach().tolist())
            loss_test += loss(prediction,target)
            
            if i_batch>5:
                break
            
        loss_test /=n_batch
    # print(labels_pred, labels_true)
    print("set true:", set(labels_true), "set pred:", set(labels_pred))
    print("synth-lead true: ",sum(1 for i in labels_true if i == 9))
    print("synth-lead pred: ",sum(1 for i in labels_pred if i == 9))
    acc = accuracy_score(labels_true, labels_pred)
    print("Accuracy:",acc)
    display_labels = list(dataset.instrument_id_map.keys())
    cm = confusion_matrix(labels_true, labels_pred)
    # print("vocals right: ", cm[10,10])
    # fig = plt.figure(figsize=(50, 50))
    # ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, normalize='true', 
    #                                         cmap=plt.cm.Blues, 
    #                                         xticks_rotation=45)
    # # disp.plot(cmap=plt.cm.Blues, ax=ax)
    # plt.savefig(pathlib.Path(eval_out_dir,'conf_matrix.png'))
    
    
    plot_confusion_matrix_2(cm,
                      display_labels,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=False,
                      save_dir=eval_out_dir)
    
    # pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model', '-m', action='store', type=pathlib.Path,
                        required=True, help='Model to be evaluated')
    parser.add_argument('--dataset', '-d', action='store', type=pathlib.Path, 
                        required=False, help='Location of the dataset')
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
    
    evaluate_class_accuracy(trained_model=model, dataset_dir=dataset_dir, gpu_id=gpu_id)