from pathlib import Path
from typing import Collection, Self
import torch.nn as nn
import torch
from torch import Tensor
import sys
from typing import List, Collection
import string
from collections import Counter

sys.path.insert(1, '/Users/lixiaoying/Desktop/Uebung_01-Primer/src')

from nlpds.abc.ex1.primer import (
    AbstractBiGramGenerator,
    AbstractBinaryLanguageClassifier,
    AbstractLanguageClassificationDataset,
    BiGram,
)

def process_sentence(sentence, vocabulary):
    #print(f"Vocabulary: {vocabulary}")
    # Assume some processing that might filter sentences
    processed = ' '.join([bi for bi in (sentence[i:i+2] for i in range(len(sentence)-1)) if bi in vocabulary])

    bgg = BiGramGenerator.from_vocabulary(vocabulary)
    indices = [bgg[bi_gram] for bi_gram in vocabulary]
    tensor = bgg.forward(sentence)
    actual = [float(tensor[idx]) for idx in indices]
    #print(f"Processing sentence: '{sentence}' -> '{actual}'")  # Debug output
    return processed

class BinaryLanguageClassifier(AbstractBinaryLanguageClassifier):
    def __init__(self,num_features: int):
        """
               Initialize the BinaryLanguageClassifier with a given number of features.
               Args:
                   num_features (int): The number of features for the classifier.
               """
        super().__init__()
        self._num_features = num_features
        self._weights = nn.Parameter(torch.randn(num_features))
        self._bias = nn.Parameter(torch.randn(1))
        self._device = torch.device('cpu')


    @property
    def num_features(self) -> int:
        """
            Get the number of features for the classifier.
            Returns:
                int: The number of features.
            """
        return self._num_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
              Perform a forward pass of the classifier.
              Args:
                  features (torch.Tensor): A tensor containing the input features.
              Returns:
                  torch.Tensor: The output of the classifier.
              """

        return (torch.matmul(features, self._weights.unsqueeze(-1)) + self._bias).squeeze(-1)

    @property
    def weights(self) -> Tensor:
        """
               Get the weights of the classifier.
               Returns:
                   Tensor: The weights of the classifier.
               """
        return self._weights

    @weights.setter
    def weights(self, weights: Tensor):
        """
                Set the weights of the classifier.
                Args:
                    weights (Tensor): The new weights for the classifier.
                """
        self._weights = nn.Parameter(weights)

    @property
    def bias(self) -> Tensor:
        """
                Get the bias of the classifier.
                Returns:
                    Tensor: The bias of the classifier.
                """
        return self._bias

    @bias.setter
    def bias(self, bias: Tensor):
        """
                Set the bias of the classifier.
                Args:
                    bias (Tensor): The new bias for the classifier.
                """
        self._bias = nn.Parameter(bias)

    @property
    def device(self) -> torch.device:
        """
               Get the device on which the classifier is running.
               Returns:
                   torch.device: The device of the classifier.
               """
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
        """
                Set the device for the classifier.
                Args:
                    device (str | torch.device): The new device for the classifier.
                """
        self._device = torch.device(device)
        self.to(self._device)

    def to(self, device: str | torch.device) -> Self:
        """
        Move the classifier to the specified device.
        Args:
            device (str | torch.device): The device to move the classifier to.
        Returns:
            Self: The classifier instance.
        """
        super().to(device)
        self.device = device
        return self

    @classmethod
    def with_num_features(cls, num_features: int) -> Self:
        """
                Create an instance of BinaryLanguageClassifier with the specified number of features.
                Args:
                    num_features (int): The number of features for the classifier.
                Returns:
                    BinaryLanguageClassifier: An instance of BinaryLanguageClassifier.
                """
        return cls(num_features)



class LanguageClassificationDataset(AbstractLanguageClassificationDataset):
    def __init__(self, data,vocabulary):
        """
        Initializes the dataset object.
        Args:
            data (list of tuples): A list of tuples where each tuple contains a raw text string and a corresponding label.
            vocabulary (set): A set of unique double character combinations (bigrams) to be used as the model's vocabulary.

        Attributes:
            data (list of tuples): Stores the input data.
            char_to_index (dict): Maps each double character combination in the vocabulary to a unique index.
        """

        self.data = data
        # pass our vocab to the model
        self.char_to_index = {char: idx for idx, char in enumerate((sorted(vocabulary)))}

    def __getitem__(self, index):
        """
        Retrieves the data point at the specified index and processes it.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing:
                - features_tensor (torch.Tensor): A tensor of relative frequencies of character pairs.
                - label_tensor (torch.Tensor): A tensor containing the label.
        """
        raw_text, label = self.data[index]

        # Split raw_text by space to get double character combinations
        char_pairs = raw_text.split()
        pair_counts = Counter(char_pairs)
        total_pairs = sum(pair_counts.values())

        # Initialize frequencies with zeros
        frequencies = [0] * len(self.char_to_index)
        # Initialize frequencies with zeros

        # Calculate the relative frequency for each pair in the vocabulary
        for pair, count in pair_counts.items():
            if pair in self.char_to_index:
                id = self.char_to_index[pair]
                frequency = count / total_pairs if total_pairs > 0 else 0
                frequencies[id] = frequency

        # Create a tensor from frequencies, using float dtype
        features_tensor = torch.tensor(frequencies, dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)  # assuming binary labels

        return features_tensor, label_tensor
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    @classmethod
    def from_files(cls, file_de: Path, file_en: Path, vocabulary: Collection[str]) -> 'LanguageClassificationDataset':
        """
        Create a LanguageClassificationDataset from files containing German and English sentences.
        Args:
            file_de (Path): Path to the file containing German sentences.
            file_en (Path): Path to the file containing English sentences.
            vocabulary (Collection[str]): A collection of bi-grams to be used as vocabulary.
        Returns:
            LanguageClassificationDataset: An instance of LanguageClassificationDataset.
        """
        # Read files and process sentences
        german_sentences = file_de.read_text(encoding='utf-8').splitlines()
        english_sentences = file_en.read_text(encoding='utf-8').splitlines()


        data = []
        for sentence in german_sentences:
            #print(f"German sentence: {sentence}")
            data.append((process_sentence(sentence, vocabulary), 0))  # 0 for German

        for sentence in english_sentences:
            #print(f"English sentence: {sentence}")
            data.append((process_sentence(sentence, vocabulary), 1))  # 1 for English
        print(f"Total samples processed: {len(data)}")  # Debugging output
        return cls(data, vocabulary)



class BiGramGenerator(AbstractBiGramGenerator):
    def __init__(self, vocabulary: Collection[str]):
        """
        Initialize the BiGramGenerator with a given vocabulary.
        Args:
            vocabulary (Collection[str]): A collection of bi-grams to be used as vocabulary.
        """
        self.vocab = list(vocabulary)
        self.index = {bigram: idx for idx, bigram in enumerate(self.vocab)}

    def bi_grams(self, sentence: str) -> List[str]:
        """
               Generate bi-grams from the given sentence.
               Args:
                   sentence (str): The sentence to generate bi-grams from.
               Returns:
                   List[str]: A list of bi-grams present in the sentence and vocabulary.
               """
        # Generate bi-grams from the sentence
        trimmed_sentence = sentence.strip()  # Remove leading and trailing whitespace
        return [trimmed_sentence[i:i+2] for i in range(len(trimmed_sentence) - 1) if trimmed_sentence[i:i+2] in self.vocab]

    @classmethod
    def from_vocabulary(cls, vocabulary: Collection[str]):
        """
             Create a BiGramGenerator instance from a given vocabulary.
                Args:
                    vocabulary (Collection[str]): A collection of bi-grams to be used as vocabulary.
                Returns:
                    BiGramGenerator: An instance of BiGramGenerator.
                """
        return cls(vocabulary)

    def forward(self, sentence: str) -> torch.Tensor:
        """
               Create a normalized frequency vector of bi-grams from the given sentence.
               Args:
                   sentence (str): The sentence to generate the bi-gram frequency vector from.
               Returns:
                   torch.Tensor: A tensor containing the normalized frequencies of bi-grams.
               """
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
        """
               Get the index of a given bi-gram in the vocabulary.
               Args:
                   bi_gram (str): The bi-gram whose index is to be found.
               Returns:
                   int: The index of the bi-gram in the vocabulary.
               Raises:
                   ValueError: If the bi-gram is not of length 2.
                   KeyError: If the bi-gram is not found in the vocabulary.
               """
        if len(bi_gram) != 2:
            raise ValueError("BiGram must be of length 2")
        if bi_gram not in self.index:
            raise KeyError(f"BiGram '{bi_gram}' not found in vocabulary")
        return self.index[bi_gram]

    def __len__(self) -> int:
        """
               Get the size of the vocabulary.
               Returns:
                   int: The number of bi-grams in the vocabulary.
               """
        return len(self.vocab)

    def __contains__(self, value: str) -> bool:
        """
               Check if a bi-gram is in the vocabulary.
               Args:
                   value (str): The bi-gram to check.
               Returns:
                   bool: True if the bi-gram is in the vocabulary, False otherwise.
               """
        return value in self.index

    @property
    def vocabulary(self) -> Collection[str]:
        """
               Get the vocabulary of bi-grams.
               Returns:
                   Collection[str]: The collection of bi-grams in the vocabulary.
               """
        return self.vocab

    def get(self, bi_gram: str, default=-1) -> int:
        """
                Get the index of a bi-gram, or return a default value if the bi-gram is not found.
                Args:
                    bi_gram (str): The bi-gram whose index is to be found.
                    default (int, optional): The default value to return if the bi-gram is not found. Defaults to -1.
                Returns:
                    int: The index of the bi-gram in the vocabulary, or the default value if not found.
                """
        if len(bi_gram) != 2:
            return default
        return self.index.get(bi_gram, default)
