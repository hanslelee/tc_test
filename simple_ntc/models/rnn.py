import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=.3,
    ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)

        return y


# import torch.nn as nn

# class RNNClassifier(nn.Module):

#     def __init(
#             self,
#             input_size, #not determined by me
#             word_vec_size,
#             hidden_size,
#             n_classes, #몇개의 클래스로 분류할거니
#             n_layers=4,
#             dropout_p=.3
#     ):
#         self.input_size = input_size  # vocabulary_size
#         self.word_vec_size = word_vec_size
#         self.hidden_size = hidden_size
#         self.n_classes = n_classes
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p

#         super().__init__()

#         self.emb = nn.Embedding(input_size, word_vec_size)
#         self.rnn = nn.LSTM(
#             input_size=word_vec_size,
#             hidden_size=hidden_size,
#             num_layers=n_layers,
#             dropout=dropout_p,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.generator = nn.Linear(hidden_size * 2, n_classes)
#         self.activation = nn.LogSoftmax(dim=-1) # Softmax + CrossEntropy 대신 LogSoftmax + NLLLoss 사용

#     def forward(self,x):
#         # |x| = (batch_size, length)
#         x = self.emb(x) 
#         # |x| = (batch_size, length, word_vec_size)
#         x, _ = self.rnn(x) 
#         # |x| = (batch_size, length, hidden_size * 2)
#         y = self.activation(self.generator(x[:, -1])) # 마지막 아웃풋만 가져온다.
#         # |y| = (batch_size, n_classes)

#         return y