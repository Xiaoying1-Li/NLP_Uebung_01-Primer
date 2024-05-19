from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Self, TypeAlias, TypeGuard

import sys
sys.path.insert(1,'/Users/lixiaoying/Desktop/Uebung_01-Primer/src')
BiGram: TypeAlias = str
def is_bi_gram(s: str) -> bool:
    """
    检查一个字符串是否是一个有效的双字母组。
    """
    return len(s) == 2 and s.isalpha()
sentence = " aaa abba bbb "
def bi_grams(sentence: str) -> list[BiGram]:
    bi_grams_list = []
    for i in range(len(sentence) - 1):
        bi_gram = sentence[i:i + 2]
        if is_bi_gram(bi_gram):
            bi_grams_list.append(bi_gram)
    return bi_grams_list


result = bi_grams(sentence)
print(result)