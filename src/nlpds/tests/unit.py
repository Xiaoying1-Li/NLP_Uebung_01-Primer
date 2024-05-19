import unittest
from pathlib import Path

import torch
import sys
sys.path.insert(1,'/Users/lixiaoying/Desktop/Uebung_01-Primer/src')

class SmallUnitTests(unittest.TestCase):
    vocab = ("aa", "ab", "ba", "xy")

    def test_bi_grams(self):
        from nlpds.submission.ex1.primer import BiGramGenerator

        bgg = BiGramGenerator.from_vocabulary(self.vocab)

        actual = bgg.bi_grams(" aaa abba bbb ")
        expected = ["aa", "aa", "ab", "ba"]
        self.assertEqual(actual, expected)

        indices = [bgg[bi_gram] for bi_gram in self.vocab]
        tensor = bgg.forward(" aaa abba bbb ")
        actual = [float(tensor[idx]) for idx in indices]
        expected = [0.5, 0.25, 0.25, 0]
        self.assertEqual(actual, expected)

    def test_dataset(self):
        from nlpds.submission.ex1.primer import LanguageClassificationDataset

        lcd = LanguageClassificationDataset.from_files(
            Path("/Users/lixiaoying/Desktop/Uebung_01-Primer/src/nlpds/tests/data/a.txt"),
            Path("/Users/lixiaoying/Desktop/Uebung_01-Primer/src/nlpds/tests/data/b.txt"),
            self.vocab,
        )
        features, labels = zip(*iter(lcd))
        self.assertEqual(len(lcd), 3, "Expected 3 samples in the dataset")
        self.assertEqual(len(features), 3, "Expected 3 samples in the dataset")
        self.assertEqual(len(labels), 3, "Expected 3 samples in the dataset")

        self.assertEqual(sum(labels), 1, "Expected 1 English sample (label=1)")

    def test_classifier(self):
        from nlpds.submission.ex1.primer import BinaryLanguageClassifier

        blc = BinaryLanguageClassifier.with_num_features(4)
        blc.weights = torch.tensor(
            [[1.0, 1.0, -1.0, -1.0]],
            requires_grad=True,
        )
        blc.bias = torch.tensor(
            [0.0],
            requires_grad=True,
        )

        tensor = blc.forward(torch.tensor([[2, 1, 1, 0]], dtype=torch.float32))
        self.assertTrue(
            (shape := tensor.flatten().view((1, -1)).shape) in {(1, 1), (1, 2)},
            f"Expected a single binary prediction, got: {shape}",
        )
        self.assertEqual(float(tensor), 2.0)


if __name__ == "__main__":
    unittest.main()

