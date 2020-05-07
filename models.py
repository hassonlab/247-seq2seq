import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Convolutional Models
class PITOM(nn.Module):
    """ A 1D convolutional classifier model.
    Args:
        num_classes: number of tokens in target vocabulary.
        num_electrodes: number of channels in electrode data (default=64).
    """
    def __init__(self, num_classes, num_electrodes=64):
        super(PITOM, self).__init__()

        self.conv1 = nn.Conv1d(num_electrodes,
                               out_channels=128,
                               kernel_size=9,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.max1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=128,
                               out_channels=128,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=128,
                               kernel_size=8,
                               stride=1,
                               padding=0)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.dropout(self.bn1(torch.relu(self.conv1(x))), p=0.2, inplace=True)
        x = self.max1(x)
        x = F.dropout(self.bn2(torch.relu(self.conv2(x))), p=0.2, inplace=True)
        x = F.dropout(self.bn3(torch.relu(self.conv3(x))), p=0.2, inplace=True)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze()
        x = self.fc(x)
        return x


class ConvNet10(nn.Module):
    """ A 2D convolutional classifier model.
    Args:
        num_classes: number of tokens in target vocabulary.
    """
    def __init__(self, num_classes):
        super(ConvNet10, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(128)

        self.conv11 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)

        # self.gpool = nn.MaxPool2d(4, kernel_size=())
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.bn5(torch.relu(self.conv5(x)))
        x = self.bn6(torch.relu(self.conv6(x)))
        x = self.bn7(torch.relu(self.conv7(x)))
        x = self.bn8(torch.relu(self.conv8(x)))
        x = self.bn9(torch.relu(self.conv9(x)))
        x = self.bn10(torch.relu(self.conv10(x)))
        x = self.bn11(torch.relu(self.conv11(x)))
        x = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze()
        x = self.fc1(x)
        return x


# Transformer Models
class MeNTALEncoderLayer(nn.Module):
    """ MeNTAL Convolutional Self-Attention Encoder Layer. """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MeNTALEncoderLayer, self).__init__()
        # Implementation of Convolutional Self-Attention
        self.conv1 = nn.Conv1d(d_model,
                               out_channels=d_model,
                               kernel_size=9,
                               stride=1,
                               padding=4,
                               bias=False)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        super(MeNTALEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: mask for src keys per batch (optional).
        """
        q, k = src.transpose(0,
                             1).transpose(1,
                                          2), src.transpose(0,
                                                            1).transpose(1, 2)
        q, k = self.conv1(q), self.conv1(k)
        q, k = q.transpose(1, 2).transpose(0,
                                           1), k.transpose(1,
                                                           2).transpose(0, 1)
        src2 = self.self_attn(q,
                              k,
                              src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MeNTALmini(nn.Module):
    """ A transformer encoder-classifier model.
    Args:
        num_electrodes: number of channels in electrode data (default=64).
        num_classes: number of tokens in target vocabulary (default=64).
        d_model: number of expected input features (default=512).
        nhead: number of heads in the multiheadattention models (default=8).
        num_encoder_layers: number of sub-encoder-layers in encoder (default=6).
        dim_feedforward: dimension of feedforward network model (default=2048).
        dropout: dropout value (default=0.1).
        activation: activation function used in encoder (default=relu).
    """
    def __init__(self,
                 num_electrodes=64,
                 num_classes=64,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super(MeNTALmini, self).__init__()

        self.model_type = 'Transformer'
        self.name = 'MeNTALmini'
        self.d_model = d_model
        self.nhead = nhead

        self.brain_encoder = nn.Linear(num_electrodes, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        """ self.mental_layer = MeNTALEncoderLayer(d_model, nhead, dim_feedforward,
                                               dropout)
        self.mental_layer_norm = nn.LayerNorm(d_model) """
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers,
                                             encoder_norm)

        self.classifier = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def forward(self, src):
        """ Encode and classify src sequence. """

        src = self.brain_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        pos_mask = (torch.triu(
            torch.ones(src.size(0), src.size(0),
                       device=src.device)) == 1).transpose(0, 1)
        pos_mask = pos_mask.float().masked_fill(pos_mask == 0,
                                                float('-inf')).masked_fill(
                                                    pos_mask == 1, float(0.0))
        """ memory = self.mental_layer_norm(
            self.mental_layer(src, src_mask=pos_mask)) """
        output = self.encoder(src)
        output = output.mean(0)
        output = self.classifier(output)

        return output

    def _reset_parameters(self):
        """ Initialize model parameters. """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.brain_encoder.bias.data.zero_()
        self.classifier.bias.data.zero_()


class MeNTAL(nn.Module):
    """ A transformer encoder-decoder model.
    Args:
        num_electrodes: number of channels in electrode data (default=64).
        num_tokens: number of tokens in target vocabulary (default=64).
        d_model: number of expected input features (default=512).
        nhead: number of heads in the multiheadattention models (default=8).
        num_encoder_layers: number of sub-encoder-layers in encoder (default=6).
        dim_feedforward: dimension of feedforward network model (default=2048).
        dropout: dropout value (default=0.1).
        activation: activation function used in encoder (default=relu).
    """
    def __init__(self,
                 num_electrodes=64,
                 num_tokens=64,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super(MeNTAL, self).__init__()

        self.model_type = 'Transformer'
        self.name = 'MeNTAL'
        self.d_model = d_model
        self.nhead = nhead

        self.brain_encoder = nn.Linear(num_electrodes, d_model)
        """ self.conv1 = nn.Conv1d(num_electrodes,
                               out_channels=d_model,
                               kernel_size=33,
                               stride=1,
                               padding=16,
                               bias=False)
        self.conv2 = nn.Conv1d(d_model,
                               out_channels=d_model,
                               kernel_size=17,
                               stride=1,
                               padding=8,
                               bias=False)
        self.conv3 = nn.Conv1d(d_model,
                               out_channels=d_model,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False) """
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.eng_encoder = nn.Linear(num_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers,
                                             encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers,
                                             decoder_norm)

        self.classifier = nn.Linear(d_model, num_tokens)

        self._reset_parameters()

    def encode(self, src):
        """ Encode source sequence.
        Args:
            src: the sequence to the encoder (required).
        """

        src = self.brain_encoder(src) * math.sqrt(self.d_model)
        """src = F.dropout(torch.relu(self.conv1(src.transpose(1, 2))),
                        p=0.02,
                        inplace=True)
        src = F.dropout(torch.relu(self.conv2(src)), p=0.02, inplace=True)
        src = F.dropout(torch.relu(self.conv3(src)), p=0.02,
                        inplace=True).transpose(1, 2) """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        memory = self.encoder(src)

        return memory

    def decode(self, memory, trg, trg_pos_mask, trg_pad_mask):
        """ Decode memory and trg sequence.
        Args:
            memory: the sequence from the encoder (required).
            trg: the sequence to the decoder (required).
            trg_pos_mask: the additive positional mask (required).
            trg_pad_mask: the multiplicative padding mask (required).
        """

        trg = self.eng_encoder(trg.float()) * math.sqrt(self.d_model)
        trg = self.pos_encoder(trg.transpose(0, 1))
        output = self.decoder(trg,
                              memory,
                              tgt_mask=trg_pos_mask.squeeze(),
                              tgt_key_padding_mask=trg_pad_mask)
        output = self.classifier(output).transpose(0, 1)

        return output

    def forward(self, src, trg, trg_pos_mask, trg_pad_mask, trg_y, criterion):
        """ Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            trg: the sequence to the decoder (required).
            trg_pos_mask: the additive positional mask (required).
            trg_pad_mask: the multiplicative padding mask (required).
            trg_y: the sequence used for loss computation (required for
                   nn.DataParallel batch splitting to work).
            criterion: the loss function for optimization (required for
                   nn.DataParallel batch splitting to work)
        """

        memory = self.encode(src)
        output = self.decode(memory, trg, trg_pos_mask, trg_pad_mask)

        # Perform loss computation and backward pass in forward call
        # for parallelism
        loss = criterion(output.contiguous().view(-1, output.size(-1)),
                         trg_y.contiguous().view(-1))
        if self.training:
            loss.backward()

        return output, trg_y, loss.unsqueeze(-1)

    def _reset_parameters(self):
        """ Initialize model parameters. """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        """###self.brain_encoder.bias.data.zero_()"""
        self.classifier.bias.data.zero_()


# Model Utilities
class PositionalEncoding(nn.Module):
    """ A positional encoding module. """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
