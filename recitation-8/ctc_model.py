import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import Levenshtein as L
from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from warpctc_pytorch import CTCLoss


DIGITS_MAP = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class DigitsModel(nn.Module):

    def __init__(self):
        super(DigitsModel, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=(1, 1), bias=False),
            nn.ELU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.ELU()
        )
        self.rnns = nn.ModuleList()
        dim = 288
        hidden_size = 256
        self.rnns.append(nn.LSTM(input_size=dim, hidden_size=hidden_size))
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=10 + 1)

    def forward(self, features):
        # features: n, 1, h, w
        embedding = self.embed(features)
        n, c, h, w = embedding.size()
        embedding = embedding.view(n, c*h, w).permute(2, 0, 1)
        # embed: t, n, f
        h = embedding
        for l in self.rnns:
            h, _ = l(h)
        logits = self.output_layer(h)
        lengths = torch.zeros((n,)).fill_(w)
        return logits, lengths

class CTCCriterion(CTCLoss):
    def forward(self, prediction, target):
        acts = prediction[0]
        act_lens = prediction[1].int()
        label_lens = prediction[2].int()
        labels = (target + 1).view(-1).int()
        return super(CTCCriterion, self).forward(
            acts=acts,
            labels=labels.cpu(),
            act_lens=act_lens.cpu(),
            label_lens=label_lens.cpu()
        )


class ER:

    def __init__(self):
        self.label_map = [' '] + DIGITS_MAP
        self.decoder = CTCBeamDecoder(
            labels=self.label_map,
            blank_id=0
        )

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        logits = prediction[0]
        feature_lengths = prediction[1].int()
        labels = target + 1
        logits = torch.transpose(logits, 0, 1)
        logits = logits.cpu()
        probs = F.softmax(logits, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)

        pos = 0
        ls = 0.
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.label_map[l] for l in labels[pos:pos + 10])
            #print("Pred: {}, True: {}".format(pred, true))
            pos += 10
            ls += L.distance(pred, true)
        assert pos == labels.size(0)
        return ls / output.size(0)

def run_eval(model, test_dataset):
    error_rate_op = ER()
    loader = DataLoader(test_dataset, num_workers=5, shuffle=False, batch_size=100)
    predictions = []
    feature_lengths = []
    labels = []
    for data_batch, labels_batch in loader:
        if torch.cuda.is_available():
            data_batch = data_batch.cuda()
        predictions_batch, feature_lengths_batch = model(data_batch.unsqueeze(1))
        predictions.append(predictions_batch.to("cpu"))
        feature_lengths.append(feature_lengths_batch.to("cpu"))
        labels.append(labels_batch.cpu())
    predictions = torch.cat(predictions, dim=1)
    labels = torch.cat(labels, dim=0)
    feature_lengths = torch.cat(feature_lengths, dim=0)
    error = error_rate_op((predictions, feature_lengths), labels.view(-1))
    return error


def run():
    best_eval = None
    epochs = 20
    batch_size = 32
    model = DigitsModel()
    model = model.cuda() if torch.cuda.is_available() else model

    labels = torch.Tensor(np.load('dataset/labels.npy')).type(torch.LongTensor)
    data = torch.Tensor(np.load('dataset/data.npy'))

    # 80/20 train/val split
    cutoff = int(0.8 * len(labels))
    dataset = TensorDataset(data[:cutoff], labels[:cutoff])
    evalset = TensorDataset(data[cutoff:], labels[cutoff:])
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    height, width = data.size()[1:]
    label_sequence_length = labels.size()[-1]

    target_lengths = torch.zeros((batch_size,)).fill_(label_sequence_length)
    ctc = CTCCriterion()
    for e in range(epochs):
        epoch_loss = 0
        for data_batch, label_batch in loader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data_batch = data_batch.cuda()
            logits, input_lengths = model(data_batch.unsqueeze(1))
            loss = ctc.forward((logits, input_lengths, target_lengths), label_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (e+1) % 5 == 0:
            with torch.no_grad():
                avg_ldistance = run_eval(model, evalset)
            if best_eval is None or best_eval > avg_ldistance:
                best_eval = avg_ldistance
                torch.save(model.state_dict(), "models/checkpoint.pt")
            print("Eval: {}".format(avg_ldistance))
        print("Loss: {}".format(epoch_loss / batch_size))

def main():
    run()


if __name__ == '__main__':
    main()
