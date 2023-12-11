import torch
import torch.nn.functional as F


def create_mlp(in_size, hidden_layers):
    return MLP(in_size, hidden_layers)

class MLP(torch.nn.Module):
    def __init__(self, in_size, h_sizes):

        super(MLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()

        first_layer = torch.nn.Sequential(
            torch.nn.Linear(in_size, h_sizes[0]),
            torch.nn.ReLU()
        )
        self.hidden_layers.append(first_layer)

        for k in range(1, len(h_sizes)-2):
            subsequent_layer = torch.nn.Sequential(
                torch.nn.Linear(h_sizes[k], h_sizes[k+1]),
                torch.nn.ReLU()
            )
            self.hidden_layers.append(subsequent_layer)
        self.hidden_layers.append(torch.nn.Linear(h_sizes[-2], h_sizes[-1]))

    def forward(self, input):
        x = input
        for layer in self.hidden_layers:
            x = layer(x)
        return x