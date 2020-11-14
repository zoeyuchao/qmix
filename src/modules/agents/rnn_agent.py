import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        if use_orthogonal:
            if use_ReLU:
                active_func = nn.ReLU()
                init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()
                init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        else:
            if use_ReLU:
                active_func = nn.ReLU()
                init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()
                init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))

        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)
    
    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self._layer_N = args.layer_N
        self._use_ReLU = args.use_ReLU
        self._use_orthogonal = args.use_orthogonal

        #self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            
        self.mlp = MLPLayer(input_shape, args.rnn_hidden_dim, self._layer_N, self._use_orthogonal, self._use_ReLU) 
        
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),1)
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0),1)
            
        self.norm = nn.LayerNorm(args.rnn_hidden_dim)
       
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.mlp.fc1[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        
        x = self.mlp(inputs)       
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h1 = self.norm(h)
        q = self.fc2(h1)
        return q, h
