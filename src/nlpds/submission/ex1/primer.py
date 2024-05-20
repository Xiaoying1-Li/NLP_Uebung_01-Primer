from pathlib import Path
from typing import Collection, Self
import torch.nn as nn
import torch
from torch import Tensor
import sys
from typing import List, Collection
import string

sys.path.insert(1, '/Users/lixiaoying/Desktop/Uebung_01-Primer/src')

from nlpds.abc.ex1.primer import (
    AbstractBiGramGenerator,
    AbstractBinaryLanguageClassifier,
    AbstractLanguageClassificationDataset,
    BiGram,
)
'''
def process_sentence(sentence, vocabulary):
    # Assume some processing that might filter sentences
    processed = ' '.join([bi for bi in (sentence[i:i+2] for i in range(len(sentence)-1)) if bi in vocabulary])
    print(f"Processing sentence: '{sentence}' -> '{processed}'")  # Debug output
    return processed
'''
def process_sentence(sentence, vocabulary):
    #print(f"Vocabulary: {vocabulary}")
    # Assume some processing that might filter sentences
    processed = ' '.join([bi for bi in (sentence[i:i+2] for i in range(len(sentence)-1)) if bi in vocabulary])

    bgg = BiGramGenerator.from_vocabulary(vocabulary)




    indices = [bgg[bi_gram] for bi_gram in vocabulary]
    tensor = bgg.forward(sentence)
    actual = [float(tensor[idx]) for idx in indices]




    #print(f"Processing sentence: '{sentence}' -> '{actual}'")  # Debug output
    return actual

class BinaryLanguageClassifier(AbstractBinaryLanguageClassifier):
    def __init__(
        self,num_features: int
        # ...
    ):
        super().__init__()
        self._num_features = num_features
        self._weights = nn.Parameter(torch.randn(num_features))
        self._bias = nn.Parameter(torch.randn(1))
        self._device = torch.device('cpu')


    @property
    def num_features(self) -> int:
        return self._num_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(1) != self._weights.size(0):
            self._weights = nn.Parameter(torch.randn(features.size(1)))
        return (torch.matmul(features, self._weights.unsqueeze(-1)) + self._bias).squeeze(-1)

    @property
    def weights(self) -> Tensor:
        return self._weights

    @weights.setter
    def weights(self, weights: Tensor):
        self._weights = nn.Parameter(weights)

    @property
    def bias(self) -> Tensor:
        return self._bias

    @bias.setter
    def bias(self, bias: Tensor):
        self._bias = nn.Parameter(bias)

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
        self._device = torch.device(device)
        self.to(self._device)

    def to(self, device: str | torch.device) -> Self:
        super().to(device)
        self.device = device
        return self

    @classmethod
    def with_num_features(cls, num_features: int) -> Self:
        return cls(num_features)
    # TODO: Document all methods from AbstractBinaryLanguageClassifier


class LanguageClassificationDataset(AbstractLanguageClassificationDataset):
    def __init__(self, data):
        self.data = data
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(string.ascii_lowercase)}  # +1 to reserve 0 for padding

    def __getitem__(self, index):
        #print(index)
        raw_text, label = self.data[index]
        #print(f"Raw text: '{raw_text}'")
        #print(f"Label: {label}")

        # Convert raw_text to string
        if isinstance(raw_text, list):
            raw_text = [str(num) for num in raw_text]
            raw_text = ''.join(raw_text)

        # Convert characters to indices
        indices = [self.char_to_index.get(char, 0) for char in raw_text.replace(" ", "")]  # remove spaces and convert

        # Create a tensor from indices, using long dtype for indices
        features_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor([label], dtype=torch.float32)  # assuming binary labels

        return features_tensor, label_tensor
    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_files(cls, file_de: Path, file_en: Path, vocabulary: Collection[str]) -> 'LanguageClassificationDataset':
        # Read files and process sentences
        german_sentences = file_de.read_text(encoding='utf-8').splitlines()
        english_sentences = file_en.read_text(encoding='utf-8').splitlines()

        # Adjust the number of sentences to process
        #german_sentences = german_sentences[:2]  # Process at most 2 German sentences
        #english_sentences = english_sentences[:1]  # Process at most 1 English sentence

        data = []
        for sentence in german_sentences:
            #print(f"German sentence: {sentence}")  # 输出德语句子
            data.append((process_sentence(sentence, vocabulary), 1))  # 0 for German

        for sentence in english_sentences:
            #print(f"English sentence: {sentence}")  # 输出英语句子
            data.append((process_sentence(sentence, vocabulary), 0))  # 1 for English


        print(f"Total samples processed: {len(data)}")  # Debugging output
        return cls(data)

    # TODO: Document all methods from AbstractLanguageClassificationDataset


class BiGramGenerator(AbstractBiGramGenerator):
    def __init__(self, vocabulary: Collection[str]):
        self.vocab = list(vocabulary)
        self.index = {bigram: idx for idx, bigram in enumerate(self.vocab)}

    def bi_grams(self, sentence: str) -> List[str]:
        # Generate bi-grams from the sentence
        trimmed_sentence = sentence.strip()  # Remove leading and trailing whitespace
        return [trimmed_sentence[i:i+2] for i in range(len(trimmed_sentence) - 1) if trimmed_sentence[i:i+2] in self.vocab]

    @classmethod
    def from_vocabulary(cls, vocabulary: Collection[str]):
        return cls(vocabulary)

    def forward(self, sentence: str) -> torch.Tensor:
        # Create a zero vector of the size of the vocabulary
        vector = torch.zeros(len(self.vocab), dtype=int)
        for bigram in self.bi_grams(sentence):
            if bigram in self.index:
                vector[self.index[bigram]] += 1
        # Normalize the vector to get frequencies
        total_bi_grams = vector.sum().item()
        if total_bi_grams > 0:
            vector = vector.float() / total_bi_grams
        return vector

    def __getitem__(self, bi_gram: str) -> int:
        if len(bi_gram) != 2:
            raise ValueError("BiGram must be of length 2")
        if bi_gram not in self.index:
            raise KeyError(f"BiGram '{bi_gram}' not found in vocabulary")
        return self.index[bi_gram]

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, value: str) -> bool:
        return value in self.index

    @property
    def vocabulary(self) -> Collection[str]:
        return self.vocab



    def get(self, bi_gram: str, default=-1) -> int:
        if len(bi_gram) != 2:
            return default
        return self.index.get(bi_gram, default)
    # TODO: Document all methods from AbstractBiGramGenerator