import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 filters,
                 filter_size,
                 embedding=None,
                 update_embedding=False,
                 input_dropout_p=0):
        super(EncoderCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        filters = [int(x) for x in filters.split(',')]
        convs = []
        for filter in filters:
            convs.append(nn.Conv2d(1, filter_size, (filter, embedding_size)))
        self.convs = nn.ModuleList(convs)

    def forward(self, input_var):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        batch, width, height = embedded.shape
        embedded = embedded.view((batch, 1, width, height))  # (batch, 1, sentence_length, embed_dim)
        outputs = []
        for conv in self.convs:
            x = F.relu(conv(embedded))  # (batch, kernel_num, H_out, 1)
            x = x.squeeze(3)  # x: (batch, kernel_num, H_out)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            outputs.append(x)
        out = torch.cat(outputs, -1)
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
        filters = MODEL_session.get('filters', '3,4,5')
        filter_size = int(MODEL_session.get('filter_size', 100))

        self.n_label = n_label
        self.n_filters = len([int(x) for x in filters.split(',')])
        self.encoder = EncoderCNN(len(vocab),
                                  embedding_size,
                                  filters,
                                  filter_size,
                                  update_embedding=update_embedding,
                                  input_dropout_p=input_dropout_p,
                                  embedding=vocab.vectors)

        self.dropout = nn.Dropout(dropout_p)
        self.predictor = nn.Linear(self.n_filters*filter_size, n_label)

    def forward(self, x, lengths=None):
        out = self.encoder(x)
        out = self.dropout(out)
        out = self.predictor(out)
        out = out.view(-1, self.n_label)
        return out
