
import torch
from model_zoo.model_utils import create_mlp

class IntentFormerDecoder(torch.nn.Module):

    def __init__(self, hidden_dim, future_horizon, num_segments, params):

        super(IntentFormerDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.future_horizon = future_horizon
        self.num_segments = num_segments
        self.params = params
        in_size = params["traj_input_dim"]
        num_hiddens = params["num_hiddens"]
        nhead = params["n_head"]
        nlayer = params["num_transformer_decoder_layers"]
        num_intent_modes = params["num_intent_modes"]
        self.num_motion_modes = num_intent_modes
        self.out_coord_dim = params["out_coord_dim"]
        coord_dim = self.out_coord_dim
        time_horizon = self.future_horizon
        self.nlayer = nlayer

        anchor_endpoints = torch.rand((self.num_motion_modes, 2))# torch.Tensor([[0, 4], [0, 2], [0, 0], [0, -2], [0, -4]])
        self.traj_anchors = anchor_endpoints.cuda()
        out_dim = time_horizon * coord_dim
        self.anchor_encoder_mlp = create_mlp(2, [hidden_dim, hidden_dim])

        self.traj_decoder_mlp = create_mlp(hidden_dim, [hidden_dim, hidden_dim, out_dim])
        self.weight_decoder_mlp = create_mlp(hidden_dim, [hidden_dim, hidden_dim, 1])

        self.transformer_decoder_layers = torch.nn.ModuleList()

        for _ in range(nlayer):
            self.transformer_decoder_layers.append(
                torch.nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
            )


    def forward(self, input):
        """
        :param input: Tensor of shape [B, T, D], with T large
        :return:
        """

        # Split the input into segments

        B, nseg, hidden_dim = input.shape

        memory = input
        anchor_encoding = self.anchor_encoder_mlp(self.traj_anchors)
        tgt = anchor_encoding[None].repeat(B, 1, 1)
        for layer in self.transformer_decoder_layers:
            x = layer(tgt, memory)
            tgt = x

        output_traj = self.traj_decoder_mlp(x)
        output_traj = output_traj.reshape(B, self.num_motion_modes, self.future_horizon, self.out_coord_dim)

        output_weight = self.weight_decoder_mlp(x)
        output_weight = output_weight[..., 0]


        return output_traj, output_weight


