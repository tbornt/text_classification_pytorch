import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from models.rnn_classifier import EncoderRNN


class RCNNTextClassifier(nn.Module):
    """
    RCNN for sentences classification.
    """
    def __init__(self, vocab, MODEL_session):
        super(RCNNTextClassifier, self).__init__()
        clf_rnn_type = MODEL_session['rnn_type']
        embedding_size = int(MODEL_session['embedding_size'])
        hidden_size = int(MODEL_session['hidden_size'])
        n_label = int(MODEL_session['n_label'])
        bidirectional = MODEL_session.get('bidirectional', False)
        max_len = int(MODEL_session.get('max_len', 500))
        n_encoder_layer = int(MODEL_session.get('n_encoder_layer', 1))
        update_embedding = MODEL_session.get('update_embedding', False)
        input_dropout_p = float(MODEL_session.get('input_dropout_p', 0.0))
        dropout_p = float(MODEL_session.get('dropout_p', 0.0))
        vocab_size = len(vocab)
        embedding = vocab.vectors

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding

        self.encoder = EncoderRNN(len(vocab),
                                  rnn_cell=clf_rnn_type,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  embedding_size=embedding_size,
                                  max_len=max_len,
                                  hidden_size=hidden_size,
                                  n_layers=n_encoder_layer,
                                  variable_lengths=True,
                                  bidirectional=bidirectional,
                                  embedding=vocab.vectors,
                                  update_embedding=update_embedding)

        self.W2 = nn.Linear(2 * hidden_size + embedding_size, hidden_size)
        self.predictor = nn.Linear(hidden_size, n_label)

    def forward(self, seq, lengths):
        embeded = self.embedding(seq)  # embeded.shape=(batch_size, num_sequences, embedding_size)

        output, _ = self.encoder(seq, lengths) # output.shape=(batch, num_sequences, 2 * embedding_size)
        fw = output[:, :, :self.hidden_size]
        bw = output[:, :, self.hidden_size:]
        x = torch.cat((fw, embeded, bw), 2)  # equation(3)
        y2 = F.tanh(self.W2(x))
        y2 = y2.permute(0, 2, 1)
        y3 = F.max_pool1d(y2, y2.size(2))
        y3 = y3.squeeze(2)
        out = self.predictor(y3)
        return out