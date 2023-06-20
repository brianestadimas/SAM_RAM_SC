import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange

# Define semantic encoder and decoder models
class SemanticCommunicationChannel(nn.Module):
    def __init__(self):
        super(SemanticCommunicationChannel, self).__init__()
        self.encoder = SemanticEncoder()
        self.decoder = SemanticDecoder()

    def forward(self, x):
        # Encode the input text into a latent representation
        encoded = self.encoder(x)

        # Send the encoded representation through the channel (no-op in this example)
        transmitted = encoded

        # Decode the transmitted representation back into the semantic space
        decoded = self.decoder(transmitted)

        return decoded


class SemanticEncoder(nn.Module):
    def __init__(self):
        super(SemanticEncoder, self).__init__()

        # Define encoder architecture using Vision Transformer
        self.patch_size = 16
        self.embed_dim = 512
        self.num_heads = 8
        self.num_layers = 6

        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, (224 // self.patch_size) ** 2 + 1, self.embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads), num_layers=self.num_layers)

    def forward(self, x):
        # Perform encoding
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(224 / self.patch_size), w=int(224 / self.patch_size))

        return x


class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

        # Define decoder architecture using Vision Transformer
        self.patch_size = 16
        self.embed_dim = 512
        self.num_heads = 8
        self.num_layers = 6

        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.num_heads), num_layers=self.num_layers)
        self.positional_embedding = nn.Parameter(torch.zeros(1, (224 // self.patch_size) ** 2 + 1, self.embed_dim))
        self.patch_embed = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=3, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # Perform decoding
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.positional_embedding
        x = self.transformer_decoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(224 / self.patch_size), w=int(224 / self.patch_size))
        x = self.patch_embed(x)

        return x.sum(dim=0).unsqueeze(0)
