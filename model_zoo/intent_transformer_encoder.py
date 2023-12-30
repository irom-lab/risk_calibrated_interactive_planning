
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

    def forward(self, input, pos_history):
        """
        :param input: Tensor of shape [B, T, D], with T large
        :return:
        """

        # Split the input into segments

        B, T, D = input.shape
        num_segments = self.num_segments

        input_reshaped = input
        encoded_input = self.local_mlp(input_reshaped)

        pos_embed = self.get_pos_sineembed(pos_history)
        transformer_encoder_input = torch.cat((encoded_input, pos_embed), dim=1)
        transformer_embedding = self.transformer_encoder(transformer_encoder_input)
        transformer_embedding = transformer_embedding[:, :T]

        return transformer_embedding

    def get_pos_sineembed(self, pos):
        batch_size, traj_len, num_coord = pos.shape
        denom = 10000 ** (2 * torch.arange(self.hidden_dim) / self.hidden_dim).cuda()
        denom = denom[None, None, :].repeat(batch_size, traj_len, 1)
        sineembed_x = torch.sin(pos[:, :, 0:1] / denom[:, :, ::4])
        cosembed_x = torch.cos(pos[:, :, 0:1] / denom[:, :, ::4])
        sineembed_y = torch.sin(pos[:, :, 1:2] / denom[:, :, ::4])
        cosembed_y = torch.cos(pos[:, :, 1:2] / denom[:, :, ::4])
        all_embed = torch.cat((sineembed_x, cosembed_x, sineembed_y, cosembed_y), dim=-1)
        return all_embed

