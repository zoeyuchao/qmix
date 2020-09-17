import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.use_orthogonal = args.use_orthogonal
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
        if self.use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))        

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = init_(nn.Linear(self.state_dim, self.embed_dim * self.n_agents))
            self.hyper_w_final = init_(nn.Linear(self.state_dim, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(init_(nn.Linear(self.state_dim, hypernet_embed)),
                                           nn.ReLU(),
                                           init_(nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)))
            self.hyper_w_final = nn.Sequential(init_(nn.Linear(self.state_dim, hypernet_embed)),
                                           nn.ReLU(),
                                           init_(nn.Linear(hypernet_embed, self.embed_dim)))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = init_(nn.Linear(self.state_dim, self.embed_dim))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(init_(nn.Linear(self.state_dim, self.embed_dim)),
                               nn.ReLU(),
                               init_(nn.Linear(self.embed_dim, 1)))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
