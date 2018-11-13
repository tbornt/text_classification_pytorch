import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding=None,
                 update_embedding=False,
                 max_len=1000,
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
        self.Max3_pool = nn.MaxPool2d((max_len-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((max_len-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((max_len-5+1, 1))

    def forward(self, input_var):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        batch, width, height = embedded.shape
        embedded = embedded.view((batch, 1, width, height))
        x1 = F.relu(self.conv3(embedded))
        x2 = F.relu(self.conv4(embedded))
        x3 = F.relu(self.conv5(embedded))
        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        out = torch.cat((x1, x2, x3), -1)
        out = out.view(batch, 1, -1)
        return out


class CNNTextClassifier(nn.Module):
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

        self.predictor = nn.Linear(3, n_label)

    def forward(self, x, lengths=None):
        out = self.encoder(x)
        out = self.predictor(out)
        out = out.view(-1, self.n_label)
        return out

