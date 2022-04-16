# From 

import torch
import torch.onnx
import argparse
import pathlib
from src.model import PitchExtractor, InstrumentClassifier, CombinedModel

#Function to Convert to ONNX 
def export_onnx(model, input_size, file_path, input_names='Input', output_names='Output'): 

    

    # set the model to inference mode 
    model.eval()

    dynamic_axes={}
    for input in input_names:
        dynamic_axes[input]={0 : 'batch_size'}
    for output in output_names:
        dynamic_axes[output]={0: 'batch_size'}

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(input_size, requires_grad=True) 

    # Export the model   
    torch.onnx.export(model,         # model being run 
         args=dummy_input,       # model input (or a tuple for multiple inputs) 
         f=file_path,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = input_names,   # the model's input names 
         output_names = output_names, # the model's output names 
         dynamic_axes=dynamic_axes
    )
    print(" ") 
    print('Model has been converted to ONNX')

def save_path(trained_model):
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    out_dir = pathlib.Path(pathlib.Path(__file__).parent, "onnx", model_name, model_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = pathlib.Path(out_dir, model_path.name.strip('.pth')+".onnx")
    
    return save_path


def export_onnx_pitch(trained_model):

    file_path = save_path(trained_model=trained_model)
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False
    
    # Initialize model
    n_bins=360
    model_sr = 16000
    model = PitchExtractor(model_sr, n_bins=n_bins, model_size=model_size, dropout=dropout)

    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    export_onnx(model, input_size=(100,1024), file_path=file_path, input_names=['Frames'], output_names=['PitchActivation'])
    return

def export_onnx_instclass(trained_model):

    file_path = save_path(trained_model=trained_model)

    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False

    
    # Initialize model
    model_sr = 16000
    model_size=model_size
    n_mfcc=20
    n_classes=11
    model = InstrumentClassifier(samplerate=model_sr,n_mfcc=n_mfcc, input_length=126, 
                                 n_classes=n_classes, model_size=model_size, dropout=dropout)

    # load trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # export to onnx
    export_onnx(model, input_size=(100,126,20), file_path=file_path, input_names=['MFCC'], output_names=['ClassPrediction'])
    return

def export_onnx_combined(trained_model):

    file_path = save_path(trained_model=trained_model)
    # directory to 
    model_path = pathlib.Path.resolve(trained_model)
    
    if not model_path.is_file():
        print (f"Model does not exist: {model_path}")
        return
    
    model_size = model_path.parents[0].name
    model_name = model_path.parents[1].name
    if 'dropout' in model_name:
        dropout=True
    else:
        dropout=False
    
    # Initialize model
    model_sr = 16000
    n_bins=360
    n_classes=11
    input_length = 126
    model = CombinedModel(samplerate=model_sr, n_classes=n_classes, 
                          n_bins=n_bins, input_length=input_length, 
                          model_size=model_size)


    # load trained model
    model.load_state_dict(torch.load(trained_model))
    model.eval()
    export_onnx(model, input_size=(100,1024,126), file_path=file_path, input_names=['Spectrogram'], 
                output_names=['PitchActivation','ClassPrediction'])
    return



def export_model(trained_model):
    
    model_path = pathlib.Path.resolve(trained_model)
    model_name = model_path.parents[1].name
    if 'instclass' in model_name:
        
        export_onnx_instclass(trained_model=trained_model)
        
    elif 'combined' in model_name:
        
        export_onnx_combined(trained_model=trained_model)
        
    elif 'pitch' in model_name:
        
        export_onnx_pitch(trained_model=trained_model)
    else:
        print('Not the right combination of model and dataset')
        
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model', '-m', action='store', type=pathlib.Path,
                        required=True, help='Model to be evaluated, or folder containing ')
    
    
    args = parser.parse_args()
    model = args.model

    
    # if model=='pitch':
    #     train_pitch(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    # elif model=='instr':
    #     train_instrument_classifier(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    # elif model=='comb':
    #     train_combined_model(dataset_dir=dataset_dir, model_size=model_size, dropout=dropout, gpu_id=gpu_id)
    
    if model.is_dir():
        for trained_model in model.rglob("*.pth"):
            print(trained_model)
            export_model(trained_model=trained_model)
    elif model.is_file():
            export_model(trained_model=model)

 