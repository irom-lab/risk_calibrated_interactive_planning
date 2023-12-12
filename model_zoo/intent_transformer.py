import torch
from model_zoo.intent_transformer_encoder import IntentFormerEncoder
from model_zoo.intent_transformer_decoder import IntentFormerDecoder
from model_zoo.model_utils import create_mlp

class IntentFormer(torch.nn.Module):

    def __init__(self, hidden_dim, num_segments, prediction_horizon, params={}):

        super(IntentFormer, self).__init__()

        self.name = "WindFormer"

        self.encoder = self.create_encoder(hidden_dim, num_segments, params)
        self.decoder = self.create_decoder(hidden_dim, prediction_horizon, num_segments, params)

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output

    def create_encoder(self, hidden_dim, num_segments, params):
        encoder = IntentFormerEncoder(hidden_dim, num_segments, params)
        return encoder

    def create_decoder(self, hidden_dim, future_horizon, num_segments, params):
        decoder = IntentFormerDecoder(hidden_dim, future_horizon, num_segments, params)
        return decoder


if __name__ == "__main__":

    horizon = 1000
    num_segments = 1
    coord_dim = 6
    traj_input_dim = (horizon // num_segments) * coord_dim
    num_hiddens = 3
    n_head = 8
    nlayer = 6
    num_intent = 1

    params = {"traj_input_dim": traj_input_dim,
              "num_hiddens": num_hiddens,
              "n_head": n_head,
              "num_transformer_encoder_layers": nlayer,
              "num_transformer_decoder_layers": nlayer,
              "coord_dim": coord_dim,
              "out_coord_dim": 2,
              "num_intent_modes": 6}
    input = torch.rand(8, horizon, coord_dim).cuda()
    model = IntentFormer(256, 1000, num_segments, 1, params).cuda()

    test_output, test_weight = model(input)
    print(test_output.shape)
    print(test_weight.shape)



