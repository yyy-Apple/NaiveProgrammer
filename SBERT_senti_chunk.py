import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from transformers import AutoTokenizer, AutoModel, BertModel
import numpy as np
from tqdm import tqdm
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SBERT(nn.Module):
    # BERT
    def __init__(self):
        super().__init__()
        self.tokinizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(DEVICE)
        self.linear = nn.Linear(768, 2).to(DEVICE)
        self.l2i = {'Accept': 1, 'Reject': 0}
        self.i2l = {1: 'Accept', 0: 'Reject'}
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)
        self.best_path = 'SBERT_senti_chunk.pth'

    def encode(self, text_list):
        """ encode a list of text into tensor """
        encoded_list = []
        for text in text_list:
            encoded_text = self.encode_one(text)
            encoded_list.append(encoded_text)
        encoded_list = torch.stack(encoded_list)
        return encoded_list

    def encode_one(self, text, max_len=510):
        """ encode one sequence into tensor, return shape torch.Size([768]) """
        ids = self.tokinizer.encode(text)
        if len(ids) < max_len:
            ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            encoded_text = self.bert(ids_tensor)[1].squeeze()
        else:
            batch_num = len(ids) // max_len + 1
            ids_list = []
            for i in range(batch_num):
                partial_ids = ids[i * max_len: (i + 1) * max_len]
                ids_list.append(partial_ids)

            # calculate the padding in last ids
            last_ones_len = len(ids) - max_len * (batch_num - 1)
            last_zeros_len = max_len - last_ones_len
            ids_list[-1] += last_zeros_len * [0]

            attn_mask = [[1] * max_len for i in range(batch_num)]
            attn_mask[-1] = [1] * last_ones_len + [0] * last_zeros_len

            ids_tensor = torch.tensor(ids_list, dtype=torch.long).to(DEVICE)
            attn_mask_tensor = torch.tensor(ids_list, dtype=torch.long).to(DEVICE)
            encoded_text = self.bert(input_ids=ids_tensor, attention_mask=attn_mask_tensor)[1].mean(dim=0)
        return encoded_text

    def train(self, train_file, val_file, batch_size=4, epochs=5):
        train_list = read_data(train_file)
        val_list = read_data(val_file)
        best_val_acc = -1.0
        for i in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            counter = 0
            train_generator = generate_batches(self.l2i, train_list, batch_size)
            for batch_text, batch_label in tqdm(train_generator, desc=f'Training Epoch {i}, train'):
                self.optimizer.zero_grad()
                counter += 1
                batch_input = self.encode(batch_text).to(DEVICE)
                batch_target = torch.tensor(batch_label).to(DEVICE)
                pred_target = self.linear(batch_input)
                loss = self.criterion(pred_target, batch_target)
                acc = compute_acc(pred_target, batch_target)
                running_loss += loss.item()
                running_acc += acc
                loss.backward()
                self.optimizer.step()
                if counter % 20 == 0:
                    print(f'On epoch {i}, the avg training loss is {running_loss / counter}, '
                          f'the avg training acc is {running_acc / counter}')

            running_loss = 0.0
            running_acc = 0.0
            counter = 0
            val_generator = generate_batches(self.l2i, val_list, batch_size)
            with torch.no_grad():
                for batch_text, batch_label in tqdm(val_generator, desc=f'Training Epoch {i}, val'):
                    counter += 1
                    batch_input = self.encode(batch_text).to(DEVICE)
                    batch_target = torch.tensor(batch_label).to(DEVICE)
                    pred_target = self.linear(batch_input)
                    loss = self.criterion(pred_target, batch_target)
                    acc = compute_acc(pred_target, batch_target)
                    running_loss += loss.item()
                    running_acc += acc
                running_loss = running_loss / counter
                running_acc = running_acc / counter
                if running_acc > best_val_acc:
                    self.save()
                    best_val_acc = running_acc
                print(f'On epoch {i}, the avg val loss is {running_loss}, '
                      f''f'the avg val acc is {running_acc}')

    def inference(self, test_file, output_file, batch_size=4):
        self.load()
        test_list = read_data(test_file)
        test_generator = generate_batches(self.l2i, test_list, batch_size, shuffle=False)
        counter = 0
        running_acc = 0.0
        output_list = []
        with torch.no_grad():
            for batch_text, batch_label in tqdm(test_generator):
                counter += 1
                batch_input = self.encode(batch_text).to(DEVICE)
                batch_target = torch.tensor(batch_label).to(DEVICE)
                pred_target = self.linear(batch_input)
                acc = compute_acc(pred_target, batch_target)
                output_list += pred_target.argmax(dim=-1).tolist()
                running_acc += acc
            running_acc = running_acc / counter
            print(f'The best model on test set achieves {running_acc} acc.')
        output_list = [self.i2l.get(item) for item in output_list]
        output_file = open(output_file, 'w')
        for out in output_list:
            print(out, file=output_file)
        output_file.flush()

    def save(self):
        torch.save(self.state_dict(), self.best_path)
        print(f'Model saved in {self.best_path}')

    def load(self):
        self.load_state_dict(torch.load(self.best_path))
        print(f'Loaded model from {self.best_path}')


def compute_acc(pred_target, target):
    total = target.size(0)
    pred = pred_target.argmax(dim=-1)
    correct = (pred == target).sum().item()
    return correct / total


def read_data(file):
    data_list = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(' ||| ')
            label = line[0]
            text = line[1]
            data_list.append((label, text))
    return data_list


def generate_batches(l2i, data_list, batch_size, shuffle=True):
    """ Return vector form """
    batch_num = len(data_list) // batch_size + 1
    if shuffle:
        np.random.shuffle(data_list)
    for i in range(batch_num):
        batch_data = data_list[i * batch_size: (i + 1) * batch_size]
        batch_text = [data[1] for data in batch_data]
        batch_label = [data[0] for data in batch_data]
        batch_label = [l2i.get(item) for item in batch_label]
        yield batch_text, batch_label


def main():
    sbert = SBERT()
    print("========== BEGIN TRAINING ===========")
    sbert.train('toy.train', 'toy.val')
    print("========== END TRAINING ==========")
    sbert.inference('toy.test', 'hypo.test')


if __name__ == '__main__':
    main()
