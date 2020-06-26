# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

        print('-' * 50)
        print("sent:", sen)
        print("word:", word)
        print("input:", input)
        print("target:", target)
        print("input_batch:", input_batch)
        print("target_batch:", target_batch)
        print('-' * 50)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)

        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        return output

model = NNLM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)

print("*"*50)
print("(begin) input_batch:", input_batch)
print("(begin) target_batch:", target_batch)
print("*"*50)

from pprint import pprint
pprint(word_dict)
pprint(number_dict)
print("*"*50)

input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

# Training
for epoch in range(1000):

    optimizer.zero_grad()
    output = model(input_batch)


    print("="*30)
    print("="*30)
    print("="*30)
    print("-"*30)
    print(model(input_batch))
    print("-"*30)
    print(model(input_batch))
    print("-"*30)
    print(model(input_batch))
    print("-"*30)
    print(model(input_batch))
    print("-"*30)
    print("="*30)
    print("="*30)
    print("="*30)


    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%50 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        print("OUTPUT: ", output.data, output.size())
        print("TARGET_BATCH: ", target_batch)

        predict = output.data.max(1, keepdim=True)
        print(predict)
        predict = predict[1]

        predicted_words = [number_dict[x.item()] for x in predict.squeeze()]

        input_words = [[number_dict[i.item()] for i in x] for x in input_batch]

        print("INPUTS:", input_words)
        print("PREDICTION:", predicted_words)
        print("\n\n\n")

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
# Use torch.Tensor.item() to get a Python number from a tensor containing a single value:
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
