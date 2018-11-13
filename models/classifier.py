import torch
from torch.autograd import Variable
import torch.nn as nn

from models.rnn_encoder import EncoderRNN


class RNNTextClassifier(nn.Module):
    def __init__(self, vocab, MODEL_session):
        super(RNNTextClassifier, self).__init__()
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

        if bidirectional:
            output_size = hidden_size * 2
        else:
            output_size = hidden_size
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
        self.predictor = nn.Linear(output_size, n_label)

    def forward(self, seq, lengths):
        output, _ = self.encoder(seq, lengths)
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
            len(lengths), output.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        pred = self.predictor(last_output)
        return pred
