from .embeddings import PositionalEmbedding, SegmentEmbedding
from .transformer import TransformerEncoder
# import sys
# sys.path.append('/home/mtran/BERT-pytorch-lm/bert/train')
from ..utils.pad import pad_masking

from torch import nn


def build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size):
    token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
    positional_embedding = PositionalEmbedding(max_len=max_len, hidden_size=hidden_size)
    segment_embedding = SegmentEmbedding(hidden_size=hidden_size)

    encoder = TransformerEncoder(
        layers_count=layers_count,
        d_model=hidden_size,
        heads_count=heads_count,
        d_ff=d_ff,
        dropout_prob=dropout_prob)

    bert = BERT(
        encoder=encoder,
        token_embedding=token_embedding,
        positional_embedding=positional_embedding,
        segment_embedding=segment_embedding,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size)

    return bert


def build_cmodel(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size):
    # vocab_size is flatten dimension from facemesh, hidden size represent the embedding size
    if(vocabulary_size != hidden_size):
        token_embedding = nn.Linear(vocabulary_size, hidden_size)
    else:
        token_embedding = nn.Identity()
    positional_embedding = PositionalEmbedding(max_len=max_len, hidden_size=hidden_size)
    segment_embedding = SegmentEmbedding(hidden_size=hidden_size)

    encoder = TransformerEncoder(
        layers_count=layers_count,
        d_model=hidden_size,
        heads_count=heads_count,
        d_ff=d_ff,
        dropout_prob=dropout_prob)

    bert = BERT(
        encoder=encoder,
        token_embedding=token_embedding,
        positional_embedding=positional_embedding,
        segment_embedding=segment_embedding,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size)

    return bert


class FineTuneModel(nn.Module):

    def __init__(self, pretrained_model, hidden_size, num_classes):
        super(FineTuneModel, self).__init__()

        self.pretrained_model = pretrained_model

        self.new_classification_layer = nn.Linear(hidden_size, num_classes)
        # self.pretrained_model.classification_layer = new_classification_layer

    def forward(self, inputs):
        sequence, segment = inputs
        token_predictions, classification_embedding = self.pretrained_model((sequence, segment))
        classification_outputs = self.new_classification_layer(classification_embedding)
        return classification_outputs


class BERT(nn.Module):

    def __init__(self, encoder, token_embedding, positional_embedding, segment_embedding, hidden_size, vocabulary_size):
        super(BERT, self).__init__()

        self.encoder = encoder
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.segment_embedding = segment_embedding
        if(hidden_size != vocabulary_size):
            self.token_prediction_layer = nn.Linear(hidden_size, vocabulary_size)
        else:
            self.token_prediction_layer = nn.Identity()
        self.classification_layer = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        sequence, segment = inputs
        if(len(sequence.size()) == 3):
            sequence = sequence.float()
        token_embedded = self.token_embedding(sequence)
        positional_embedded = self.positional_embedding(sequence)
        segment_embedded = self.segment_embedding(segment)
        embedded_sources = token_embedded + positional_embedded + segment_embedded

        mask = pad_masking(sequence)
        encoded_sources = self.encoder(embedded_sources, mask)
        token_predictions = self.token_prediction_layer(encoded_sources)
        classification_embedding = encoded_sources[:, 0, :]
        classification_output = self.classification_layer(classification_embedding)
        return token_predictions, classification_embedding
