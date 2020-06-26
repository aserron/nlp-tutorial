'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from pprint import pprint
import sys

dtype = torch.FloatTensor

# Text-CNN Parameter
embedding_size = 2 # n-gram
sequence_length = 3
num_classes = 2  # 0 or 1
filter_sizes = [2, 2, 2] # n-gram window
num_filters = 3 #output channels

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)

inputs = []
for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

targets = []
for out in labels:
    targets.append(out) # To using Torch Softmax Loss function

input_batch = torch.LongTensor(inputs)
target_batch = torch.LongTensor(targets)

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.num_filters_total = num_filters * len(filter_sizes)

        # W: embedding
        # --> W[word_idx] = embedding for the word
        self.W = nn.Parameter(torch.empty(vocab_size, embedding_size).uniform_(-1, 1)).type(dtype)

        # Weight: for the final, fully connected layer
        self.Weight = nn.Parameter(torch.empty(self.num_filters_total, num_classes).uniform_(-1, 1)).type(dtype)
        self.Bias = nn.Parameter(0.1 * torch.ones([num_classes])).type(dtype)
        self.conv_layers = []
        self.conv_weights = []
        for filter_size in filter_sizes:
            conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)
            self.conv_layers.append(conv) # a conv_layer's weight.size(): [3, 1, 2, 2]

            # conv_weight = nn.Parameter(torch.empty(3, 1, 2, 2).uniform_(-1, 1)).type(dtype)
            # conv.weight = conv_weight
            # self.conv_weights.append(conv_weight)

        self.conv_layers = nn.ModuleList(self.conv_layers)


    def forward(self, X):

        # X: input batch - [num_sent(=batch_size)*sen_len(=#words in a sent)]

        embedded_chars = self.W[X] # embedded_chars: [batch_size, sen_len(=#words in a sent), embedding_length]
        # W[X] operation: W[comp]이 Tensor X 내부 모든 component에 대해 수행된다고 생각하면 됨

        # for i in range(0, len(X)):
        #     for j in range(0, len(X[i])):
        #         W[X[i][j]] == embedded_chars[i][j]
        #
        #         print(i, len(X), j, len(X[i]))
        #         print("*"*20)
        #         pprint(self.W[X[i][j]])
        #         print("*"*20)
        #         pprint(embedded_chars[i][j])
        #         print("="*20)

        embedded_chars = embedded_chars.unsqueeze(1)
        # [batch_size, channel(=1), sen_len(=#words in a sent), embedding_length]

        pooled_outputs = []
        for idx, filter_size in enumerate(filter_sizes):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]

            # by using the predefined conv_layers, we can prevent the iterative initialization at every forward() call
            conv = self.conv_layers[idx](embedded_chars)

            # old way: it initializes the weights of the filters whenever the system calls the forward function
            # conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(embedded_chars)


            # filters are randomly initiallized as in Conv2d.__init__()
            # self.reset_parameters() -> init.kaiming_uniform_(self.weight, ...)

            h = F.relu(conv) # [batch_size, num_filters, n - h + 1(=len_sen - len_filter + 1), 1(=embedding_size/filter_width = #horizontal walk)]

            # print("-"*10)
            # print("FilterSize:", filter_size)
            # print(h.size())
            # print("-"*10)

            # Dim Exercise
            # m = nn.Conv2d(16, 33, (1, 100))
            # input = torch.randn(20, 16, 50, 100)
            # output = m(input)
            # print(output.size())
            # print("*&^"*10)

            # mp : ((filter_height=(n-h+1), filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1)) #sequence_length = len_sent = #words in a sent

            pooled = mp(h)
            #[batch_size(=6), output_channel(=3), output_width(=1), output_height(=1)]

            pooled = pooled.permute(0, 3, 2, 1)
            #[batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]

            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3=num_filters) * 3(=len(filter_sizes)]

        #torch.cat(Tensor, dim) - concat w.r.t the dim

        # A filter converts a sentence into a scalar. Now there are total 9 filters(=3*3), therefore, the filters convert a sent into a vector that has 9 components.

        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), num_filters_total)]

        # print("-"*10)
        # print(h_pool.size())
        # print(h_pool_flat.size())
        # print("-"*10)
        # print(h_pool)
        # print("*"*10)
        # print(h_pool_flat)
        # print("*"*10)
        # sys.exit()

        model = torch.mm(h_pool_flat, self.Weight) + self.Bias # [batch_size, num_classes]

        # Bias: broadcasted from [num_classes] to [batch_size, num_classes]
        # print("*"*30)
        # print(torch.mm(h_pool_flat, self.Weight))
        # print("*"*30)
        # print(self.Bias)
        # print("*"*30)
        # print(torch.mm(h_pool_flat, self.Weight) + self.Bias)
        # print("*"*30)

        return model

model = TextCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for name, param in model.named_parameters():
   print(name, param)

# Training
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(input_batch) # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)

    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:

        print("=" * 30)
        print(output)
        print("*" * 30)
        print(model(input_batch))
        print("*" * 30)
        print(target_batch)
        print("=" * 30)
        print(output.size())
        print("*" * 30)
        print(target_batch.size())
        print("=" * 30)

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Test
test_text = 'sorry hate you'
tests = [np.asarray([word_dict[n] for n in test_text.split()])]
test_batch = torch.LongTensor(tests)

# Predict
predict = model(test_batch).data.max(1, keepdim=True)[1]

# Practice for max dimension
# print("*"*30)
# print(model(test_batch).data)
# print("*"*30)
# print(model(test_batch).data.max(1, keepdim=True))
# print("*"*30)
# print(model(test_batch).data.max(0, keepdim=True))
# print("*"*30)
# print(model(test_batch).data.max(1, keepdim=False))
# print("*"*30)
# print(model(test_batch).data.max(0, keepdim=False))
# print("*"*30)

if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")