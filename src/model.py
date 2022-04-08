import torch
import torch.nn as nn
import torch.nn.functional as F

def get_conv1d_block(in_channels, out_channels, kernel_size, stride=1, padding=32):
    
    net = []

    net.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                stride, padding, bias=False))

    
    net.append(nn.BatchNorm1d(out_channels,momentum=0.5))
    net.append(nn.ReLU())
    net.append(nn.MaxPool1d(2))
    return nn.Sequential(*net)






class PitchExtractor(nn.Module):
    def __init__(self, sampling_rate, model_size='tiny'):
        super().__init__()

        


        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_size]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        in_channels = [1]+filters[:-1]
        out_channels = filters
        # kernel_sizes = [512, 64, 64, 64, 64, 64]
        kernel_sizes = [(512,)] + 5 * [(64,)]
        # strides = [4, 1, 1, 1, 1, 1]
        strides = [(4,)] + 5 * [(1,)]
        padding = [(256,)] + 5 * [(32,)]
        self.in_features = capacity_multiplier*64
        self.pitch_bins=360
        
        # Convolutional layer blocks
        net = []
        for l, in_c, out_c, w, s, p in zip(layers, in_channels, out_channels, kernel_sizes, strides, padding):

            net.append(get_conv1d_block(in_c, out_c, w, s, p))

        
        self.conv_layers = nn.Sequential(*net)
        
        self.classifier = nn.Sequential(
            torch.nn.Linear(self.in_features, self.pitch_bins),
            # torch.nn.Sigmoid()
        )


        

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, self.in_features)
        x= self.classifier(x)

        return x

class InstrumentClassifier(nn.Module):
    def __init__(self, sampling_rate, model_size='tiny'):
        super().__init__()
    


if __name__ == "__main__":

    

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    input = torch.randn(23,1,1024).to(device)
    x = input
    
    x = x.reshape(-1, 256)
    print(f"output rearranged: {x.size()}\nDevice: {x.device}")

    pitch_model = PitchExtractor(44100,model_size='medium').to(device)
    output = pitch_model.forward(input)
    print(f"output: {output.size()}\n")
    print(f"in_f: {pitch_model.in_features}")