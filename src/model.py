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


def get_conv2d_block(in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0,31,32)):
    
    net = []
    net.append(nn.ConstantPad2d(padding=padding,value=0.0))
    net.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride, bias=False))

    
    net.append(nn.BatchNorm2d(out_channels,momentum=0.5))
    net.append(nn.ReLU())
    net.append(nn.MaxPool2d(kernel_size=(2,1)))
    return nn.Sequential(*net)


def get_mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.ReLU())
    return nn.Sequential(*net)




class PitchExtractor(nn.Module):
    def __init__(self, samplerate, model_size='tiny'):
        super().__init__()

        


        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_size]

        layers = [1, 2, 3, 4, 5, 6]
        
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        in_channels = [1]+filters[:-1]
        out_channels = filters
        
        kernel_sizes = [(512,)] + 5 * [(64,)]
        
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
            nn.Linear(self.in_features, self.pitch_bins),
            # torch.nn.Sigmoid()
        )


        

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(-1, self.in_features)
        x= self.classifier(x)

        return x

class InstrumentClassifier(nn.Module):
    # To use with Nsyth dataset
    def __init__(self, samplerate, n_mfcc, input_length, n_classes, model_size='medium'):
        super().__init__()


        capacity_multiplier = {
            'small': 1, 'medium': 2, 'large': 4, 'large-shallow':4,
        }[model_size]

        self.input_length = input_length
        self.n_classes = n_classes

        self.gru_size = capacity_multiplier*128

        if capacity_multiplier=='large-shallow':
            self.n_layers_mlp=1
        else:
            self.n_layers_mlp = 2

        if capacity_multiplier=='large-shallow':
            self.mlp_size = capacity_multiplier*16
        else:
            self.mlp_size = capacity_multiplier*64
        
        self.gru = nn.GRU(input_size=n_mfcc, hidden_size=self.gru_size, batch_first=True)

        net = []
        net.append(get_mlp(self.gru_size*self.input_length, self.mlp_size, self.n_layers_mlp))
        net.append(nn.Linear(self.mlp_size,self.n_classes))
        self.classifier = nn.Sequential(*net)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x.reshape(-1, self.gru_size*self.input_length)
        x= self.classifier(x)

        return x


class CombinedModel(nn.Module):
    # To use with Nsyth dataset
    def __init__(self, samplerate, n_classes, n_bins, input_length, model_size='medium'):
        super().__init__()

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_size]

        self.output_conv_size = 4
        self.n_classes = n_classes
        self.pitch_bins=n_bins
        self.input_length = input_length
        self.gru_size = capacity_multiplier*16

        layers = [1, 2, 3, 4, 5, 6]
        # layers = [1, 2 ]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        in_channels = [1]+filters[:-1]
        out_channels = filters
        # kernel_sizes = [512, 64, 64, 64, 64, 64]
        kernel_sizes = [(512,1)] + 5 * [(64,1)]
        # strides = [4, 1, 1, 1, 1, 1]
        strides = [(4,1)] + 5 * [(1,1)]
        padding = [(0,0,256,256)] + 5 * [(0,0,31,32)]
        self.in_features = self.output_conv_size*filters[-1]
        
        
        # Convolutional layer blocks
        net = []
        for l, in_c, out_c, w, s, p in zip(layers, in_channels, out_channels, kernel_sizes, strides, padding):

            net.append(get_conv2d_block(in_c, out_c, w, s, p))

        
        self.conv_layers = nn.Sequential(*net)
        
        self.classifier_pitch = nn.Sequential(
            nn.Linear(self.in_features, self.pitch_bins),
            # torch.nn.Sigmoid()
        )

        

        
        self.gru = nn.GRU(input_size=self.in_features, hidden_size=self.gru_size, batch_first=True)
        
        self.classifier_inst = nn.Linear(self.gru_size*self.input_length, self.n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print('unsq input',x.shape)
        # x = self.conv_layers(x)
        for conv in self.conv_layers:
            x = conv(x)
            # print("after conv",x.shape)

        # print("after all conv",x.shape)
        x = x.permute(0,3,1,2)
        # print("after permutation x",x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print("after reshape x", x.shape)
        pred_pitch = self.classifier_pitch(x)
        # print("after classif pitch pred_pitch",pred_pitch.shape)
        pred_inst, _ = self.gru(x)
        # print("after gru pred_inst",pred_inst.shape)
        pred_inst = pred_inst.reshape(-1, self.gru_size*self.input_length)
        # print("after reshape pred_inst",pred_inst.shape)
        pred_inst = self.classifier_inst(pred_inst)
        # print("after classif inst pred_inst",pred_inst.shape)

        # print('for debug')

        return pred_pitch, pred_inst
    


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    # input = torch.randn(4096,1024,126).to(device)
    # x = input
    
    # x = x.reshape(-1, 256)
    # print(f"output rearranged: {x.size()}\nDevice: {x.device}")

    # pitch_model = PitchExtractor(44100,model_size='medium').to(device)
    # output = pitch_model.forward(input)
    # print(f"output: {output.size()}\n")
    # print(f"in_f: {pitch_model.in_features}")

    mod = 'combined'
    if mod == 'combined':
        input = torch.randn(10,1024,126)
        model = CombinedModel(16000,11,360,126)
        pitch_model = PitchExtractor(44100,model_size='medium')
        print("pytorch_total_params",sum(p.numel() for p in model.parameters()))
        print("pytorch_total_params",sum(p.numel() for p in pitch_model.parameters()))
        # model.to(device)


        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a 

        
        p_pitch, p_inst = model(input)