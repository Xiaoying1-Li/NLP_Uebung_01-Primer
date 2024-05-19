from pathlib import Path
from typing import Collection
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from collections import Counter
from itertools import chain
from nlpds.abc.ex1.primer import BiGram
from nlpds.submission.ex1.primer import (
    BiGramGenerator,
    BinaryLanguageClassifier,
    LanguageClassificationDataset,
)

if __name__ == "__main__":

  data_root = Path("/Users/lixiaoying/Desktop/Uebung_01-Primer/data/ex1/primer/")
  deu_dev = data_root / "deu_dev.txt"
  deu_test = data_root / "deu_test.txt"
  deu_train = data_root / "deu_train.txt"
  eng_dev = data_root / "eng_dev.txt"
  eng_test = data_root / "eng_test.txt"
  eng_train = data_root / "eng_train.txt"

'''
def build_vocabulary(files, n_most_common=1000):
    # Create a counter to hold bi-grams and their frequencies
    bi_gram_counter = Counter()

    # Process each file
    for file in files:
        text = file.read_text(encoding='utf-8').lower()
        words = text.split()
        bi_grams = zip(words[:-1], words[1:])
        bi_gram_counter.update(bi_grams)

    # Select the n most common bi-grams
    common_bi_grams = {bg for bg, _ in bi_gram_counter.most_common(n_most_common)}
    # Print the most common bi-grams
    #print("Most common bi-grams:")
    #for bg, count in bi_gram_counter.most_common(n_most_common):
    #    print(bg, count)
    return common_bi_grams
'''

def build_vocabulary(files, n_most_common=1000):
    # 创建一个计数器来保存 bi-gram 及其频率
    bi_gram_counter = Counter()

    # 处理每个文件
    for file in files:
        text = file.read_text(encoding='utf-8').lower()
        bi_grams = [text[i:i+2] for i in range(len(text) - 1)]
        bi_gram_counter.update(bi_grams)

    # 选择出现频率最高的前 n 个 bi-gram
    common_bi_grams = {bg for bg, _ in bi_gram_counter.most_common(n_most_common)}
    return common_bi_grams

# Use the training files to build the vocabulary
vocabulary = build_vocabulary([deu_train, eng_train])

# Extract individual datasets
train_dataset = LanguageClassificationDataset.from_files(deu_train, eng_train, vocabulary)
dev_dataset = LanguageClassificationDataset.from_files(deu_dev, eng_dev, vocabulary)
test_dataset = LanguageClassificationDataset.from_files(deu_test, eng_test, vocabulary)


def custom_collate_fn(batch):
    # 获取 batch 中每个样本的特征和标签
    features, labels = zip(*batch)
    # 计算 batch 中最长的特征长度
    max_length = max(len(feature) for feature in features)
    # 填充不同长度的特征张量
    padded_features = []
    for feature in features:
        padded_feature = torch.cat([feature, torch.zeros(max_length - len(feature), dtype=torch.long)])
        padded_features.append(padded_feature)
    # 将填充后的特征张量和标签转换为张量并返回
    return torch.stack(padded_features), torch.stack(labels)

model = BinaryLanguageClassifier(num_features=len(vocabulary))
criterion = nn.BCEWithLogitsLoss()  # Binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

for epoch in trange(num_epochs, desc="Epoch"):
    model.train()  # Set the model to training mode
    for features, labels in train_dataloader:
        print("Features shape:", features.shape)
        if features.nelement() == 0:
            raise ValueError("Features are empty. Check data processing.")
        optimizer.zero_grad()
        outputs = model(features)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

    # Evaluation on the development set
    model.eval()  # Set the model to evaluation mode
    total_loss, total_accuracy = 0, 0
    for features, labels in dev_dataloader:
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        predictions = outputs.round()  # Assuming sigmoid output
        total_accuracy += (predictions == labels).sum().item()

    # Print average loss and accuracy for the epoch
    avg_loss = total_loss / len(dev_dataloader)
    avg_accuracy = total_accuracy / len(dev_dataset)
    print(f"Epoch {epoch+1}: Dev Loss: {avg_loss:.4f}, Dev Accuracy: {avg_accuracy:.4f}")


model.eval()  # Set the model to evaluation mode
total_loss, total_accuracy = 0, 0
with torch.no_grad():  # Disable gradient computation
    for features, labels in test_dataloader:
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        predictions = outputs.round()
        total_accuracy += (predictions == labels).sum().item()

avg_loss = total_loss / len(test_dataloader)
avg_accuracy = total_accuracy / len(test_dataset)
print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")


# Save the model
torch.save(model.state_dict(), 'model_weights.pth')

# Optionally, save results to a file
with open('results.txt', 'w') as f:
    f.write(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}\n")

"""
   Training script template. Change as you see fit!

   ## Setup: Data
   data_root = Path("/Users/smile/Desktop/NLP/NLP/data/ex1/primer/")
   deu_dev = data_root / "deu_dev.txt"
   deu_test = data_root / "deu_test.txt"
   deu_train = data_root / "deu_train.txt"
   eng_dev = data_root / "eng_dev.txt"
   eng_test = data_root / "eng_test.txt"
   eng_train = data_root / "eng_train.txt"

   vocabulary: Collection[BiGram] = ...  # TODO
   train_dataset: LanguageClassificationDataset = ...  # TODO
   dev_dataset: LanguageClassificationDataset = ...  # TODO
   test_dataset: LanguageClassificationDataset = ...  # TODO

   ## Setup: Training Hyper-Parameters
   model: BinaryLanguageClassifier = ...  # TODO

   criterion = ...  # TODO

   learning_rate: float = ...  # TODO
   optimizer: torch.optim.Optimizer = ...  # TODO

   num_epochs: int = ...  # TODO
   batch_size: int = ...  # TODO

   ## Training
   dev_dataloader: DataLoader = ...  # TODO
   for epoch in trange(num_epochs, desc="Epoch"):
       train_dataloader: DataLoader = ...  # TODO

       model.train()
       for batch in train_dataloader:
           ...  # TODO

       ## Evaluation: Dev Set
       model.eval()
       with torch.no_grad():
           for batch in dev_dataloader:
               ...  # TODO

   ## Evaluation: Test Set
   test_dataloader: DataLoader = ...  # TODO

   model.eval()
   with torch.no_grad():
       ...  # TODO

   ## Evaluation: Save Results?
   ...  # TODO

 """
# Dummy implementation of creating a vocabulary and loading datasets
