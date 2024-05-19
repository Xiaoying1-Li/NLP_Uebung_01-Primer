from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Self, TypeAlias, TypeGuard

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset


class AbstractBinaryLanguageClassifier(nn.Module):
    @property
    @abstractmethod
    def num_features(self) -> int:
        """
        Returns the size of the input features.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """
        Forward pass of the classifier.
        Returns unnormalized logits for binary classification.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self) -> Tensor:
        """
        Gets the weights of the classifier.
        """
        raise NotImplementedError

    @weights.setter
    @abstractmethod
    def weights(self, weights: Tensor):
        """
        Sets the weights of the classifier.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def bias(self) -> Tensor:
        """
        Gets the bias of the classifier.
        """
        raise NotImplementedError

    @bias.setter
    @abstractmethod
    def bias(self, bias: Tensor):
        """
        Sets the bias of the classifier.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the current device.
        """
        raise NotImplementedError

    @device.setter
    @abstractmethod
    def device(self, device: str | torch.device):
        """
        Move all PyTorch modules to a given device.
        """
        raise NotImplementedError

    def to(self, device: str | torch.device) -> Self:
        """
        Moves the classifier to a given device.
        """
        super().to(device)
        self.device = torch.device(device)
        return self

    @classmethod
    @abstractmethod
    def with_num_features(cls, num_features: int) -> Self:
        """
        Create a binary language classifier with a given number of features.
        The classifier is expected to have randomly initialized weights and a bias.
        """
        raise NotImplementedError

BiGram: TypeAlias = str


def is_bi_gram(s: str) -> TypeGuard[BiGram]:
    """
    Type guard for bi-grams (str's of length 2).
    """
    return len(s) == 2


class AbstractLanguageClassificationDataset(TorchDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        Returns a tuple of features and the target label.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_files(
        cls,
        file_de: Path,
        file_en: Path,
        vocabulary: Collection[BiGram],
    ) -> Self:
        """
        Create a language classification dataset from two files,
        one containing German sentences and the other English ones,
        and a vocabulary of valid bi-grams.
        """
        raise NotImplementedError


class AbstractBiGramGenerator(ABC):
    @abstractmethod
    def bi_grams(self, sentence: str) -> list[BiGram]:
        """
        Generate bi-grams from a sentence.
        Returns a list of bi-grams (str's of length 2).
        If the sentence contains no valid bi-grams, returns an empty list.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, sentence: str) -> Tensor:
        """
        Generate a bi-gram-frequency feature vector for a sentence.
        """
        raise NotImplementedError

    def __call__(self, sentence: str) -> Tensor:
        return self.forward(sentence)

    @abstractmethod
    def __getitem__(self, bi_gram: BiGram) -> int:
        """
        Returns the index of a bi-gram.
        If the given bi-gram is not in the vocabulary, raises a KeyError.
        If the given value is not a valid bi-gram, raises a ValueError.
        """
        raise NotImplementedError

    def get(self, bi_gram: BiGram, default=-1) -> int:
        """
        Returns the index of a bi-gram.
        If the given value is not a valid bi-gram or not in the vocabulary,
        returns the default value.
        """
        return self[bi_gram] if bi_gram in self else default

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of bi-grams in the vocabulary.
        """
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, value: BiGram) -> bool:
        """
        Check if a bi-gram is in the vocabulary.
        If the given value is not a valid bi-gram, return False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vocabulary(self) -> Collection[BiGram]:
        """
        Get the vocabulary of bi-grams.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_vocabulary(cls, vocabulary: Collection[BiGram]) -> Self:
        """
        Create a bi-gram generator from a vocabulary of valid bi-grams.
        """
        raise NotImplementedError
