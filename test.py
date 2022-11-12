# import torch
# import torch.nn as nn
import numpy as np

# vocab_dict = {}
# with open('data/USPTO-50k_no_rxn/USPTO-50k_no_rxn.vocab.txt', 'r') as f:
#     for line in f.readlines():
#         vocab = line.strip().split('\t')
#         vocab_dict[vocab[0]] = int(vocab[1])
# print(vocab_dict)

# rand = np.random.rand()
#
# print(rand)


# import torch.optim as optim
# import torch.nn as nn
#
# x = torch.tensor([[1, 3, 4], [1, 1, 4]], requires_grad=True, dtype=torch.float32)
# y = torch.tensor([1, 2])
#
# model = nn.Linear(3, 50)
#
# optimizer = optim.Adam(model.parameters())
# loss_fn = torch.nn.NLLLoss(reduction='sum')
#
# model.train()
#
# optimizer.zero_grad()
# out = model(x)
# loss = loss_fn(out, y)
# loss.backward()
# optimizer.step()

# mask = torch.eq(x, 1)
# print(mask)

# masked_X = x[[0, 1, 0], [0, 1, 0]]
# y = torch.nonzero(x)
# # print(masked_X)
# y = [x[i] for i in range(2)]
# # y = masked_X[[0, 1]]
# print(y)

def get_smiles():
    with open('data/USPTO-50k_no_rxn/src-train.txt', 'r') as f:
        lines = f.readlines()
    return lines


def get_tokens(smile):
    tokens = smile.strip().split(' ')
    return tokens

smiles = get_smiles()
tokens_all = []
for smile in smiles:
    tokens = get_tokens(smile)
    tokens_all.append(tokens)
import numpy as np
np_tokens = np.array(tokens_all)
print(np_tokens)


