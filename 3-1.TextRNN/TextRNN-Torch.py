'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# TextRNN Parameter
batch_size = len(sentences)
n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell(= the dimension of the vector h)

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

        # print('-' * 50)
        # print("sent:", sen)
        # print("word:", word)
        # print("input:", input)
        # print("target:", target)
        # print("input_batch:", input_batch)
        # print("target_batch:", target_batch)
        # print('-' * 50)

    return input_batch, target_batch

# to Torch.Tensor
input_batch, target_batch = make_batch(sentences)
input_batch = torch.Tensor(input_batch)
target_batch = torch.LongTensor(target_batch)

print("*"*30)
print("input_batch.size():", input_batch.size())
print("target_batch.size()", target_batch.size())
print("len_word_dict:", len(list(word_dict.keys())))
print("*"*30)

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        # input: one-hot(dim = n_class)
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)

        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden, X):

        # X: input_batch: [batch_size, n_step, n_class]
        # hidden: [num_layers(=1) * num_directions(=1), batch_size, hidden_size(=the dimension of the vector h)]

        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]

        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : dimension preserved

        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]

        model = torch.mm(outputs, self.W) + self.b # model : [batch_size, n_class]

        # self.b is broadcasted from [n_class] to [batch_size, n_class]

        return model

model = TextRNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for name, param in model.named_parameters():
   print("PARAM: ", name, param.size())
   print("-"*30)

# Training
for epoch in range(5000):
    optimizer.zero_grad()

    # hidden : [num_layers * num_directions, batch_size, hidden_size(=the dimension of the vector h)]
    hidden = torch.zeros(1, batch_size, n_hidden)

    # input_batch : [batch_size, n_step, n_class]
    output = model(hidden, input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if ((epoch + 1) % 1000 == 0) or (epoch == 0):

        print("="*30)
        print(output)
        print("*"*30)
        print(target_batch)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        print("="*30)

    loss.backward()
    optimizer.step()

input = [[word_dict[x] for x in sen.split()[:2]] for sen in sentences]
input = [[np.eye(n_class)[x] for x in i] for i in input]
input = torch.Tensor(input)

# Predict
hidden = torch.zeros(1, batch_size, n_hidden)
predict = model(hidden, input).data.max(1, keepdim=True)[1]

print("*"*30)
print(model(hidden, input).data)
print("*"*30)
print(model(hidden, input).data.max(1, keepdim=True))
print("*"*30)
print(model(hidden, input).data.max(1, keepdim=True)[1])
print("*"*30)
print(predict.squeeze())
print("*"*30)

print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])