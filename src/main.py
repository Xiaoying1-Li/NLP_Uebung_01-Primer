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

def build_vocabulary(files, n_most_common=4):
    # 创建一个计数器来保存 bi-gram 及其频率
    bi_gram_counter = Counter()

    # 处理每个文件
    for file in files:
        text = file.read_text(encoding='utf-8').lower()
        # 去除空格
        text = text.replace(' ', '')
        # 确保bi-gram只包含两个字符，从第一个字符开始抓
        bi_grams = [text[i:i+2] for i in range(len(text) - 1)]
        bi_gram_counter.update(bi_grams)

    # 选择出现频率最高的前 n 个 bi-gram
    common_bi_grams = {bg for bg, _ in bi_gram_counter.most_common(n_most_common)}
    return common_bi_grams

# Use the training files to build the vocabulary
vocabulary = build_vocabulary([deu_train, deu_dev])
print(vocabulary)

# Extract individual datasets
train_dataset = LanguageClassificationDataset.from_files(deu_train, eng_train, vocabulary)
dev_dataset = LanguageClassificationDataset.from_files(deu_dev, eng_dev, vocabulary)
test_dataset = LanguageClassificationDataset.from_files(deu_test, eng_test, vocabulary)


def custom_collate_fn(batch):
    features, labels = zip(*batch)
    max_length = 1000
    padded_features = []
    for feature in features:
        padded_feature = torch.cat([feature, torch.zeros(max_length - len(feature), dtype=torch.long)])
        padded_features.append(padded_feature)
    return torch.stack(padded_features).float(), torch.stack(labels).float().squeeze(1)



model = BinaryLanguageClassifier(num_features=len(vocabulary))
print(f"Model initialized with {model.num_features} features.")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")  # 打印初始模型参数以检查其初始状态

criterion = nn.BCEWithLogitsLoss()  # Binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
batch_size = 1000

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

for epoch in trange(num_epochs, desc="Epoch"):
    model.train()  # Set the model to training mode
    for features, labels in train_dataloader:
        #print("Features shape:", features.shape)
        if features.nelement() == 0:
            raise ValueError("Features are empty. Check data processing.")
        optimizer.zero_grad()
        outputs = model(features)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        #print(f"Epoch {epoch + 1}, Loss: {loss.item()}")  # 输出每个batch的损失值
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
        #print(f"Outputs: {outputs}")
        #print(f"Labels: {labels}")
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        predictions = outputs.round()
        total_accuracy += (predictions == labels).sum().item()

avg_loss = total_loss / len(test_dataloader)
avg_accuracy = total_accuracy / len(test_dataset)
print(avg_accuracy)
print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: { avg_accuracy:.4f}")


# Save the model
torch.save(model.state_dict(), 'model_weights.pth')

# Optionally, save results to a file
with open('results.txt', 'w') as f:
    f.write(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}\n")


# Dummy implementation of creating a vocabulary and loading datasets
