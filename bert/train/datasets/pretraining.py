from bert.preprocess import PAD_INDEX, MASK_INDEX, CLS_INDEX, SEP_INDEX

from tqdm import tqdm

from random import random, randint
import numpy as np
import os
import pandas as pd

class IndexedCorpus:
    def __init__(self, data_path, dictionary, dataset_limit=None):
        self.indexed_documents = []
        with open(data_path) as file:
            for document in tqdm(file):
                indexed_document = []
                for sentence in document.split('|'):
                    indexed_sentence = []
                    for token in sentence.strip().split():
                        indexed_token = dictionary.token_to_index(token)
                        indexed_sentence.append(indexed_token)
                    if len(indexed_sentence) < 1:
                        continue
                    indexed_document.append(indexed_sentence)
                if len(indexed_document) < 2:
                    continue
                self.indexed_documents.append(indexed_document)

                if dataset_limit is not None and len(self.indexed_documents) >= dataset_limit:
                    break

    def __getitem__(self, item):
        return self.indexed_documents[item]

    def __len__(self):
        return len(self.indexed_documents)


class MaskedDocument:
    def __init__(self, sentences, vocabulary_size):
        self.sentences = sentences
        self.vocabulary_size = vocabulary_size
        self.THRESHOLD = 0.15

    def __getitem__(self, item):
        """Get a masked sentence and the corresponding target.

        For wiki-example, [5,6,MASK_INDEX,8,9], [0,0,7,0,0]
        """
        sentence = self.sentences[item]

        masked_sentence = []
        target_sentence = []

        for token_index in sentence:
            r = random()
            if r < self.THRESHOLD:  # we mask 15% of all tokens in each sequence at random.
                if r < self.THRESHOLD * 0.8:  # 80% of the time: Replace the word with the [MASK] token
                    masked_sentence.append(MASK_INDEX)
                    target_sentence.append(token_index)
                elif r < self.THRESHOLD * 0.9:  # 10% of the time: Replace the word with a random word
                    random_token_index = randint(5, self.vocabulary_size-1)
                    masked_sentence.append(random_token_index)
                    target_sentence.append(token_index)
                else:  # 10% of the time: Keep the word unchanged
                    masked_sentence.append(token_index)
                    target_sentence.append(token_index)
            else:
                masked_sentence.append(token_index)
                target_sentence.append(PAD_INDEX)

        return masked_sentence, target_sentence

    def __len__(self):
        return len(self.sentences)


class MaskedCorpus:

    def __init__(self, data_path, dictionary, dataset_limit=None):
        source_corpus = IndexedCorpus(data_path, dictionary, dataset_limit=dataset_limit)

        self.sentences_count = 0
        self.masked_documents = []
        for indexed_document in source_corpus:
            masked_document = MaskedDocument(indexed_document, vocabulary_size=len(dictionary))
            self.masked_documents.append(masked_document)

            self.sentences_count += len(masked_document)

    def __getitem__(self, item):
        return self.masked_documents[item]

    def __len__(self):
        return len(self.masked_documents)


class PairedDataset:

    def __init__(self, data_path, dictionary, dataset_limit=None):
        self.source_corpus = MaskedCorpus(data_path, dictionary, dataset_limit=dataset_limit)
        self.dataset_size = self.source_corpus.sentences_count
        self.corpus_size = len(self.source_corpus)

    def __getitem__(self, item):

        document_index = randint(0, self.corpus_size-1)
        document = self.source_corpus[document_index]
        sentence_index = randint(0, len(document) - 2)
        A_masked_sentence, A_target_sentence = document[sentence_index]

        if random() < 0.5:  # 50% of the time B is the actual next sentence that follows A
            B_masked_sentence, B_target_sentence = document[sentence_index + 1]
            is_next = 1
        else:  # 50% of the time it is a random sentence from the corpus
            random_document_index = randint(0, self.corpus_size-1)
            random_document = self.source_corpus[random_document_index]
            random_sentence_index = randint(0, len(random_document)-1)
            B_masked_sentence, B_target_sentence = random_document[random_sentence_index]
            is_next = 0

        sequence = [CLS_INDEX] + A_masked_sentence + [SEP_INDEX] + B_masked_sentence + [SEP_INDEX]

        # segment : something like [0,0,0,0,0,1,1,1,1,1,1,1])
        segment = [0] + [0] * len(A_masked_sentence) + [0] + [1] * len(B_masked_sentence) + [1]

        target = [PAD_INDEX] + A_target_sentence + [PAD_INDEX] + B_target_sentence + [PAD_INDEX]

        return (sequence, segment), (target, is_next)

    def __len__(self):
        return self.dataset_size

class PairedDatasetC:
    def __init__(self, data_path, dataset_limit=None):
        #load the whole dataset to RAM (less than 10 GB), dataset_limit refers to nfiles to load
        self.data = []
        self.len = 0
        self.n_features = 0
        for file in os.listdir(data_path):
            fdata = pd.read_csv(os.path.join(data_path, file), header=None).values
            self.n_features = fdata.shape[1]
            if(fdata.shape[0] >= 10):
                self.data.append(fdata)
                self.len += 1
                if(dataset_limit and self.len >= dataset_limit):
                    break
        self.THRESHOLD = 0.15

        # np.random.seed(0); self.CLS_INDEX = np.random.rand(1, self.n_features)
        self.SEP_INDEX = self.CLS_INDEX = self.MASK_INDEX = self.PAD_INDEX = np.zeros((1, self.n_features))
        # np.random.seed(1); self.SEP_INDEX = np.random.rand(1, self.n_features)

    def mask_helper(self, current_video):
        n_frames, n_features = current_video.shape[0], current_video.shape[1]
        split_index = np.random.randint(5, n_frames-4)
        partA, partB = current_video[:split_index, :], current_video[split_index:, :]

        masked_video_A, masked_video_B = [], []
        target_video_A, target_video_B = [], []

        for i in range(partA.shape[0]):
            r = random()
            if(r < self.THRESHOLD):
                masked_video_A.append(self.MASK_INDEX.reshape(-1,))
                target_video_A.append(partA[i, :])
            else:
                masked_video_A.append(partA[i, :])
                target_video_A.append(self.PAD_INDEX.reshape(-1,))

        for i in range(partB.shape[0]):
            r = random()
            if(r < self.THRESHOLD):
                masked_video_B.append(self.MASK_INDEX.reshape(-1,))
                target_video_B.append(partB[i, :])
            else:
                masked_video_B.append(partB[i, :])
                target_video_B.append(self.PAD_INDEX.reshape(-1,))

        return [np.array(masked_video_A),
                np.array(masked_video_B),
                np.array(target_video_A),
                np.array(target_video_B)]

    def __getitem__(self, item):
        current_video = self.data[item]
        masked_video_A, masked_video_B, target_video_A, target_video_B = self.mask_helper(current_video)

        if random() < 0.5:  # 50% of the time B is the actual next sentence that follows A
            is_next = 1
        else:  # 50% of the time it is a random sentence from the corpus
            random_video_index = randint(0, self.len-1)
            random_video = self.data[random_video_index]
            unused1, masked_video_B, unused2, target_video_B = self.mask_helper(random_video)
            is_next = 0

        #sequence = [CLS_INDEX] + masked_video_A + [SEP_INDEX] + masked_video_B + [SEP_INDEX]
        sequence = np.concatenate([self.CLS_INDEX,
                                    masked_video_A,
                                    self.SEP_INDEX,
                                    masked_video_B,
                                    self.SEP_INDEX], axis=0)
        # segment : something like [0,0,0,0,0,1,1,1,1,1,1,1])
        segment = [0] + [0] * len(masked_video_A) + [0] + [1] * len(masked_video_B) + [1]

        # target = [PAD_INDEX] + A_target_sentence + [PAD_INDEX] + B_target_sentence + [PAD_INDEX]
        target = np.concatenate([self.PAD_INDEX,
                                    target_video_A,
                                    self.PAD_INDEX,
                                    target_video_B,
                                    self.PAD_INDEX], axis=0)
        return (sequence, segment), (target, is_next)

    def __len__(self):
        return self.len
