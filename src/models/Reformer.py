import torch
import torch.nn as nn
from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import ReformerLayer
from ..layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Note: Reformer cannot accomplish cross attention, adapted here in BERT-style.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads, bucket_size=configs.bucket_size,
                                  n_hashes=configs.n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]
