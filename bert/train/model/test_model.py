from bert import build_model, build_cmodel

import torch


def test_encoder():
    model = build_model(hidden_size=512, layers_count=6, heads_count=8, d_ff=1024, dropout_prob=0.1, max_len=512,
                        vocabulary_size=100)

    example_sequence = torch.tensor([[1, 2, 3, 4, 5], [2, 1, 3, 0, 0]])
    example_segment = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

    token_predictions, classification_output = model((example_sequence, example_segment))

    batch_size, seq_len, target_vocabulary_size = 2, 5, 100
    assert token_predictions.size() == (batch_size, seq_len, target_vocabulary_size)
    print('ok.')

def test_cmodel():
    model = build_cmodel(hidden_size=64, layers_count=3, heads_count=4, d_ff=512, dropout_prob=0.1, max_len=512,
                        vocabulary_size=16)
    example_sequence = torch.randn(2, 5, 16)
    example_segment = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

    token_predictions, classification_output = model((example_sequence, example_segment))
    batch_size, seq_len, target_vocabulary_size = 2, 5, 16
    assert token_predictions.size() == (batch_size, seq_len, target_vocabulary_size)
    print('ok.')

test_encoder()
test_cmodel()
