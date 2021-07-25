from bert.preprocess import PAD_INDEX

from torch import nn
from .utils.pad import pad_masking

class MLMNSPLossModelC(nn.Module):

    def __init__(self, model):
        super(MLMNSPLossModelC, self).__init__()

        self.model = model
        self.mlm_loss_function = nn.L1Loss()
        self.nsp_loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        
        sequence, segment = inputs
        mask = pad_masking(sequence)

        mlm_outputs, nsp_outputs = outputs
        mlm_targets, is_nexts = targets

        mlm_outputs, nsp_predictions = mlm_outputs, nsp_outputs.argmax(dim=1)
        predictions = (mlm_outputs, nsp_predictions)

        batch_size, seq_len, vocabulary_size = mlm_outputs.size()
        
        mask_flat = list(mask.view(batch_size * seq_len, ).cpu()) #True if position is padded else False, pad pos are all 0
        mlm_outputs_flat = mlm_outputs.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = mlm_targets.view(batch_size * seq_len, vocabulary_size)
        
        mlm_loss = self.mlm_loss_function(mlm_outputs_flat[mask_flat], mlm_targets_flat[mask_flat]) # extract relevant frames here
        nsp_loss = self.nsp_loss_function(nsp_outputs, is_nexts)

        loss = mlm_loss + nsp_loss

        return predictions, loss.unsqueeze(dim=0)

class MLMNSPLossModel(nn.Module):

    def __init__(self, model):
        super(MLMNSPLossModel, self).__init__()

        self.model = model
        self.mlm_loss_function = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
        self.nsp_loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)

        mlm_outputs, nsp_outputs = outputs
        mlm_targets, is_nexts = targets

        mlm_predictions, nsp_predictions = mlm_outputs.argmax(dim=2), nsp_outputs.argmax(dim=1)
        predictions = (mlm_predictions, nsp_predictions)

        batch_size, seq_len, vocabulary_size = mlm_outputs.size()

        mlm_outputs_flat = mlm_outputs.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = mlm_targets.view(batch_size * seq_len)

        mlm_loss = self.mlm_loss_function(mlm_outputs_flat, mlm_targets_flat)
        nsp_loss = self.nsp_loss_function(nsp_outputs, is_nexts)

        loss = mlm_loss + nsp_loss

        return predictions, loss.unsqueeze(dim=0)


class ClassificationLossModel(nn.Module):

    def __init__(self, model):
        super(ClassificationLossModel, self).__init__()

        self.model = model
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)
        loss = self.loss_function(outputs, targets)

        return predictions, loss.unsqueeze(dim=0)
