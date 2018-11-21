import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, embedding_size, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


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
        attention = MODEL_session.get('attention', False)
        self.attention = attention
        if self.attention:
            self.attention_type = MODEL_session.get('attention_type', 'dot')

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
        if self.attention and self.attention_type == 'general':
            self.att_W = nn.Linear(output_size, output_size)

    def attention_layer(self, output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        output : contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n)

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        hidden = final_state.squeeze(0)
        if self.attention_type == 'dot':
            attn_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attn_weights = torch.bmm(self.att_W(output), hidden.unsqueeze(2)).squeeze(2)
        else:
            raise Exception('attention type: %s is not supported' % self.attention_type)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, seq, lengths):
        output, _ = self.encoder(seq, lengths)  # output.shape = (batch_size, num_seq, hidden_size)
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
            len(lengths), output.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if output.is_cuda:
            idx = idx.cuda(output.data.get_device())
        last_output = output.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        if not self.attention:
            pred = self.predictor(last_output)
        else:
            attn_out = self.attention_layer(output, last_output)
            pred = self.predictor(attn_out)
        return pred
