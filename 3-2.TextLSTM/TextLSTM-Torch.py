'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

dtype = torch.FloatTensor

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
word_dict = {n: i for i, n in enumerate(char_arr)}
number_dict = {i: w for i, w in enumerate(char_arr)}
n_class = len(word_dict) # number of class(=number of vocab)

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

# TextLSTM Parameters
n_step = 3
n_hidden = 128

def make_batch(seq_data):
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return torch.Tensor(input_batch), torch.LongTensor(target_batch)

class TextLSTM(nn.Module):

    # MOSTLY THE SAME EXCEPT FOR THE USAGE OF THE CELL STATE

    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden_state, cell_state, X):
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        # hidden_state: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # cell_state: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        # hidden_state_size: the same with the RNN - hidden state
        # cell_state_size: the same with the RNN - hidden state
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        # outputs: [n_step, batch_size, num_directions(=1) * n_hidden]
        # outputs_size: the same with the RNN - outputs

        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model

input_batch, target_batch = make_batch(seq_data)

print("*"*30)
print("input_batch_size:", input_batch.size())
print("*"*30)
print("target_batch_size:", target_batch.size())
print("*"*30)

model = TextLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for name, param in model.named_parameters():
   print("PARAM: ", name, param.size())
   print("-"*30)

# Training
for epoch in range(1000):
    optimizer.zero_grad()

    hidden_state = torch.zeros(1, len(input_batch), n_hidden)
    cell_state = torch.zeros(1, len(input_batch), n_hidden)

    output = model(hidden_state, cell_state, input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

inputs = [sen[:3] for sen in seq_data]

hidden_state_t = torch.zeros(1, len(input_batch), n_hidden)
cell_state_t = torch.zeros(1, len(input_batch), n_hidden)

predict = model(hidden_state_t, cell_state_t, input_batch).data.max(1, keepdim=True)[1]

print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])
