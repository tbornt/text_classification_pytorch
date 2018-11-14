import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding=None,
                 update_embedding=False,
                 input_dropout_p=0):
        super(EncoderCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.input_dropout = nn.Dropout(p=input_dropout_p)

        self.conv3 = nn.Conv2d(1, 1, (3, embedding_size))
        self.conv4 = nn.Conv2d(1, 1, (4, embedding_size))
        self.conv5 = nn.Conv2d(1, 1, (5, embedding_size))

    def forward(self, input_var):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        batch, width, height = embedded.shape
        embedded = embedded.view((batch, 1, width, height))  # (batch, 1, sentence_length, embed_dim)
        x1 = F.relu(self.conv3(embedded))  # (batch, kernel_num, H_out, 1)
        x2 = F.relu(self.conv4(embedded))
        x3 = F.relu(self.conv5(embedded))
        # Pooling
        x1 = x1.squeeze(3)  # x: (batch, kernel_num, H_out)
        x2 = x2.squeeze(3)
        x3 = x3.squeeze(3)

        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        out = torch.cat((x1, x2, x3), -1)
        return out


class CNNTextClassifier(nn.Module):
    """
    Convolutional Neural Networks for Sentence Classification in PyTorch 
    https://arxiv.org/abs/1408.5882
    """
    def __init__(self, vocab, MODEL_session):
        super(CNNTextClassifier, self).__init__()
        embedding_size = int(MODEL_session['embedding_size'])
        update_embedding = MODEL_session.get('update_embedding', False)
        input_dropout_p = float(MODEL_session.get('input_dropout_p', 0.0))
        dropout_p = float(MODEL_session.get('dropout_p', 0.0))
        n_label = int(MODEL_session['n_label'])

        self.n_label = n_label
        self.encoder = EncoderCNN(len(vocab),
                                  embedding_size,
                                  update_embedding=update_embedding,
                                  input_dropout_p=input_dropout_p,
                                  embedding=vocab.vectors)

        self.dropout = nn.Dropout(dropout_p)
        self.predictor = nn.Linear(3, n_label)

    def forward(self, x, lengths=None):
        out = self.encoder(x)
        out = self.dropout(out)
        out = self.predictor(out)
        out = out.view(-1, self.n_label)
        return out
