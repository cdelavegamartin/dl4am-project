import netron
import argparse
import pathlib



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
        for trained_model in model.rglob("*.onnx"):
            print(trained_model)
            netron.start(str(trained_model))
    elif model.is_file():
            netron.start(str(model))
