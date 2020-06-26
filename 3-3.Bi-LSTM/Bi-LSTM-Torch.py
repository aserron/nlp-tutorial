'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F

dtype = torch.FloatTensor

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}

special_token = "<PAD>"
special_token_idx = len(word_dict)
word_dict[special_token] = special_token_idx
number_dict[special_token_idx] = special_token

n_class = len(word_dict)
max_len = len(sentence.split()) #max_len = 27
n_hidden = 5

print("*"*30)
print(len(sentence), type(sentence))
print("*"*30)
print(len(sentence.split(" ")), max_len)
print("*"*30)
print(word_dict)
print("*"*30)
print(number_dict)
print("*"*30)

def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):

        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [special_token_idx] * (max_len - len(input)) # padding by 0
        target = word_dict[words[i + 1]]

        # print("-"*30)
        # print("input:", input)
        # print("len_input:", len(input))
        # print("target:", target)
        # print("-"*30)
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


def hard_prediction(model, initial_input):

    dynamic_input = torch.empty_like(initial_input)
    dynamic_input.data = initial_input.clone()
    print("INITIAL:", dynamic_input.data.max(2)[1].squeeze())
    num_tokens = 1
    while num_tokens < len(dynamic_input[0]):
        hidden_state = torch.zeros(1 * 2, len(dynamic_input), n_hidden)
        cell_state = torch.zeros(1 * 2, len(dynamic_input), n_hidden)
        predict = model(hidden_state, cell_state, dynamic_input).data.max(1, keepdim=True)[1]
        predict = predict.squeeze()

        # print("*"*30)
        # print("num_tokens:", num_tokens)
        # print("predict:", predict)
        # print(dynamic_input.data.max(2)[1])
        # print("*"*30)

        dynamic_input[0][num_tokens] = torch.FloatTensor(np.eye(n_class)[predict.item()])
        num_tokens += 1

    print("Final:", dynamic_input.data.max(2)[1])

    token_output = [number_dict[x.item()] for x in dynamic_input.data.max(2)[1].squeeze()]

    return token_output


def easy_prediction(model, input_batch):
    hidden_state = torch.zeros(1 * 2, len(input_batch), n_hidden)
    cell_state = torch.zeros(1 * 2, len(input_batch), n_hidden)
    predict = model(hidden_state, cell_state, input_batch).data.max(1, keepdim=True)[1]
    easy_prediction = ['Lorem'] + [number_dict[n.item()] for n in predict.squeeze()]

    return easy_prediction


class BiLSTM(nn.Module):

    # Compared to vanilla LSTM, The only difference is the dimension of the hidden state (and therefore the cell state.
        # there are two tensors of [1, batch_size, n_hidden], each of which can be understood as the hidden units for a direction.

    # Some details are important for the indexing of the output, but it can be easily understood that the index of the output is based on the corresponding input words or the index of the corresponding input words.

    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.randn([n_hidden * 2 * 2, n_class]).type(dtype)) # n_hidden * num_direction * first_and_end_of_the_outputs
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden_state, cell_state, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        # outputs: [n_step, batch_size, num_directions(=2) * n_hidden]

        # OLD WAY
        # outputs = outputs[-1]  # [batch_size, num_directions(=2) * n_hidden]

        # BUT is not it making more sense?: Yes, indeed. The loss decreases much much faster than the old way.
        outputs = torch.cat([outputs[0], outputs[-1]], 1)

        # torch.cat(Tensor, dim) - concat w.r.t the dim

        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model

input_batch, target_batch = make_batch(sentence)

model = BiLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10000):
    optimizer.zero_grad()

    hidden_state = torch.zeros(1 * 2, len(input_batch), n_hidden)
    cell_state = torch.zeros(1 * 2, len(input_batch), n_hidden)

    output = model(hidden_state, cell_state, input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:

        # print("*"*30)
        # print("input_batch_size: ", input_batch.size())
        # print("target_batch_size: ", target_batch.size())
        # print("output_size: ", output.size())
        # print("hidden_state_size: ", hidden_state.size())
        # print("cell_state_size: ", cell_state.size())
        # print("*"*30)
        # print("output:", output)
        # print("target:", target_batch)
        print("*"*30)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        print("BEFORE:", input_batch[0].max(1)[1])
        print("HARD PREDICTION:", hard_prediction(model, torch.Tensor(input_batch[0]).unsqueeze(0)))
        print("EASY PREDICTION:", easy_prediction(model, input_batch))
        print("ANSWER:", sentence.split(" "))

        print("*"*30)

    loss.backward()
    optimizer.step()
