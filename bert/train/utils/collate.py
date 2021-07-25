# from bert.preprocess import PAD_INDEX
import numpy as np

def pretraining_collate_function(batch):

    targets = [target for _, (target, is_next) in batch]
    longest_target = max(targets, key=lambda target: len(target))
    max_length = len(longest_target)

    padded_sequences = []
    padded_segments = []
    padded_targets = []
    is_nexts = []
    PAD_INDEX = 0
    for (sequence, segment), (target, is_next) in batch:
        length = len(sequence)
        padding = [PAD_INDEX] * (max_length - length)
        padded_sequence = sequence + padding
        padded_segment = segment + padding
        padded_target = target + padding

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        padded_targets.append(padded_target)
        is_nexts.append(is_next)

    count = 0
    for target in targets:
        for token in target:
            if token != PAD_INDEX:
                count += 1

    return (padded_sequences, padded_segments), (padded_targets, is_nexts), count
    
def pretraining_collate_functionC(batch):
    
    targets = [target for _, (target, is_next) in batch]
    longest_target = max(targets, key=lambda target: len(target))
    max_length = len(longest_target)

    padded_sequences = []
    padded_segments = []
    padded_targets = []
    is_nexts = []

    n_features = longest_target.shape[1]
    for (sequence, segment), (target, is_next) in batch:
        length = len(sequence)
        padding = np.zeros((max_length - length, n_features))  
        padded_sequence = np.concatenate([sequence, padding], axis=0) 
        padded_segment = segment + [0] * (max_length - length)
        padded_target = np.concatenate([target, padding], axis=0)

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        padded_targets.append(padded_target)
        is_nexts.append(is_next)

    count = 0
    for target in targets:
        for token in target:
            if token.sum() != 0:
                count += 1

    return (np.array(padded_sequences), np.array(padded_segments)), (np.array(padded_targets), is_nexts), count


def classification_collate_function(batch):

    lengths = [len(sequence) for (sequence, _), _ in batch]
    max_length = max(lengths)

    padded_sequences = []
    padded_segments = []
    labels = []

    for (sequence, segment), label in batch:
        length = len(sequence)
        padding = [PAD_INDEX] * (max_length - length)
        padded_sequence = sequence + padding
        padded_segment = segment + padding

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        labels.append(label)

    count = len(labels)

    return (padded_sequences, padded_segments), labels, count

def classification_collate_functionC(batch):

    lengths = [len(sequence) for (sequence, _), _ in batch]
    max_length = max(lengths)

    padded_sequences = []
    padded_segments = []
    labels = []

    for (sequence, segment), label in batch:
        length = len(sequence)
        padding = np.zeros((max_length - length, n_features))  
        padded_sequence = np.concatenate([sequence, padding], axis=0) 
        padded_segment = segment + [0] * (max_length - length)

        padded_sequences.append(padded_sequence)
        padded_segments.append(padded_segment)
        labels.append(label)

    count = len(labels)

    return (np.array(padded_sequences), np.array(padded_segments)), labels, count