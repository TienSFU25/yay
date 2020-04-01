import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

import argparse
import numpy as np
import pdb
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from data import *
from sklearn.model_selection import train_test_split
# from model import Encoder, Decoder

MAX_LENGTH= 60
SOS_token = '<SOS>'
EOS_token = '<EOS>'

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size=16, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers*2, inputs.size(0), self.hidden_size))
        context = Variable(torch.zeros(self.n_layers*2, inputs.size(0), self.hidden_size))

        return hidden, context

    def forward(self, input, last_hidden):
        # hidden = self.init_hidden(input)
        embedded = self.embedding(input)

        # [16, 60, 64]
        # [2, 16, 64]
        output, next_hidden = self.lstm(embedded, last_hidden)

        return output, next_hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, slot_size, intent_size, embedding_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)

        # just concat the 2 vectors
        lstm_input_size = self.embedding_size + self.embedding_size * 2
        self.lstm = nn.LSTM(lstm_input_size, self.hidden_size, self.n_layers, batch_first=True)
        # self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.slot_out = nn.Linear(self.hidden_size, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size, self.intent_size)

    def init_hidden(self, target):
        hidden = Variable(torch.zeros(self.n_layers, target.size(0), self.hidden_size))
        context = Variable(torch.zeros(self.n_layers, target.size(0), self.hidden_size))

        return hidden, context

    def forward(self, input, last_hidden, aligned_encoder_output):
        # input = B * 1 (16 * 1)
        # B * D => B * 1 * D
        aligned = aligned_encoder_output.unsqueeze(1)
        embedded = self.embedding(input)

        # B * 1 * (3D) (16 * 1 * 192)
        concatted = torch.cat((aligned, embedded), 2)
        # concatted = _concatted.squeeze(1)

        # B * 1 * D / 1 * B * D
        _lstm_output, next_hidden = self.lstm(concatted, last_hidden)
        lstm_output = _lstm_output.squeeze(1)
        slot_output = self.slot_out(lstm_output)
        softmaxed = F.log_softmax(slot_output, dim=1)

        return softmaxed, next_hidden

# B * 60 -> B * 60
def trainny(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, decoder_start_token, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden(input_tensor)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(1)

    loss = 0
    encoder_outputs, new_encoder_hidden = encoder(input_tensor, encoder_hidden)
    # decoder_hidden = new_encoder_hidden
    decoder_hidden = decoder.init_hidden(target_tensor)
    decoder_input = Variable(torch.LongTensor([[decoder_start_token]*config.batch_size])).transpose(1, 0)

    aligned = encoder_outputs.transpose(0, 1)
    # slot_scores_raw = []

    for di in range(target_length):
        # decoder_input = encoder_outputs[di]
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, aligned[di])
        # pdb.set_trace()

        topv, topi = decoder_output.topk(1)
        decoder_input = topi.detach()  # detach from history as input

        # 1 * slot_size / 1 * 1
        batch_target = target_tensor[:, di]
        loss += criterion(decoder_output, batch_target)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def eval_single(input_tensor, target_tensor, encoder, decoder):
    criterion = nn.NLLLoss()

def evaluate(test_data, encoder, decoder, decoder_start_token, max_length=MAX_LENGTH):
    loss = 0

    for iter, batch in enumerate(getBatch(config.batch_size, test_data)):
        x, y_1, y_2 = zip(*batch)
        input_tensor = torch.cat(x)
        target_tensor = torch.cat(y_1)

        # loss += trainny(input_tensor, target_tensor, encoder, decoder, decode_start_token)

    print("Total loss", loss)

def train(config):    
    # train_data, word2index, tag2index, intent2index = preprocessing(config.file_path, config.max_length)
    all_data, word_labeler, tag_labeler, intent_labeler = preprocessing(config.file_path, config.max_length)

    num_words = len(word_labeler.classes_)
    num_slots = len(tag_labeler.classes_)
    num_intent = len(intent_labeler.classes_)

    encoder1 = EncoderRNN(num_words, config.embedding_size, config.hidden_size)
    # slot_size, intent_size, embedding_size, hidden_size
    attn_decoder1 = AttnDecoderRNN(num_slots, num_intent, config.embedding_size, config.hidden_size)
    decode_start_token = tag_labeler.transform([SOS])[0]

    pct = int(len(all_data) * 0.7)
    train_data = all_data[:pct]
    test_data = all_data[pct:]

    trainIters(encoder1, attn_decoder1, 50, train_data, decode_start_token, print_every=500)
    # evaluate(test_data, encoder1, attn_decoder1, decode_start_token)

def trainIters(encoder, decoder, n_iters, train_data, decode_start_token, print_every=1, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.NLLLoss()

    for iter in range(n_iters):
        for _, batch in enumerate(getBatch(config.batch_size, train_data)):
            x, y_1, y_2 = zip(*batch)
            input_tensor = torch.cat(x)
            target_tensor = torch.cat(y_1)

            loss = trainny(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, decode_start_token)

            print_loss_total += loss

        # if iter % print_every == 0:
        # print_loss_avg = print_loss_total / print_every
        print("Iteration", iter, "loss", print_loss_total)
        print_loss_total = 0

    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)    

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob', help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()
    train(config)
