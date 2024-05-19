import unittest
from pathlib import Path
import sys
sys.path.insert(1,'/Users/lixiaoying/Desktop/Uebung_01-Primer/src')

class TestImplementedAbstract(unittest.TestCase):
    def __isabstractmethod(self, cls, method: str):
        with self.subTest(f"{cls.__name__}.{method}"):
            self.assertTrue(hasattr(cls, method))
            self.assertFalse(
                getattr(getattr(cls, method), "__isabstractmethod__", False),
                f"{cls.__name__}.{method} is @abstractmethod"
            )

    def test_bgg(self):
        from nlpds.submission.ex1.primer import BiGramGenerator as bgg

        for method in (
            "__call__",
            "__contains__",
            "__getitem__",
            "__len__",
            "bi_grams",
            "forward",
            "from_vocabulary",
            "get",
            "vocabulary",
        ):
            self.__isabstractmethod(bgg, method)

    def test_blc(self):
        from nlpds.submission.ex1.primer import BinaryLanguageClassifier as blc

        for method in (
            "bias",
            "forward",
            "num_features",
            "weights",
            "with_num_features",
        ):
            self.__isabstractmethod(blc, method)

    def test_lcd(self):
        from nlpds.submission.ex1.primer import LanguageClassificationDataset as lcd

        for method in (
            "__getitem__",
            "__len__",
            "from_files",
        ):
            self.__isabstractmethod(lcd, method)


if __name__ == "__main__":
    unittest.main()


