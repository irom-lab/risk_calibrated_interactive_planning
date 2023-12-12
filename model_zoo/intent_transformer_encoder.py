
import torch
from model_zoo.model_utils import create_mlp

class IntentFormerEncoder(torch.nn.Module):

    def __init__(self, hidden_dim, num_segments, params):

        super(IntentFormerEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.params = params
        in_size = params["traj_input_dim"]
        num_hiddens = params["num_hiddens"]
        nhead = params["n_head"]
        nlayer = params["num_transformer_encoder_layers"]

        self.local_mlp = create_mlp(in_size, hidden_layers=[hidden_dim]*num_hiddens)
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                                          batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=nlayer)

    def forward(self, input):
        """
        :param input: Tensor of shape [B, T, D], with T large
        :return:
        """

        # Split the input into segments

        B, T, D = input.shape
        num_segments = self.num_segments

        input_reshaped = input
        encoded_input = self.local_mlp(input_reshaped)

        transformer_embedding = self.transformer_encoder(encoded_input)

        return transformer_embedding
