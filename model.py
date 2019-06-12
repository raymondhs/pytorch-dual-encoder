import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self, num_embeddings, padding_idx, embed_dim=320, hidden_size=512, num_layers=1, bidirectional=False, dropout=0.1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths, enforce_sorted=True):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist(), enforce_sorted=enforce_sorted)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ], dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return sentemb


class DualEncoder(nn.Module):
    def __init__(self, num_embeddings, padding_idx=0, embed_dim=320, 
                 hidden_size=512, num_layers=1, bidirectional=True,
                 dropout=0.1, enforce_sorted_source=True):
        super().__init__()
        
        self.encoder = Encoder(num_embeddings=num_embeddings,
                               padding_idx=padding_idx,
                               embed_dim=embed_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)
        self.enforce_sorted_source = enforce_sorted_source
        output_units = hidden_size
        if bidirectional:
            output_units *= 2
        self.M = nn.Parameter(torch.Tensor(output_units, output_units))
        nn.init.normal_(self.M)
    
    def forward(self, x_source, x_source_lengths, x_target, x_target_lengths):
        source_embeddings = self.encoder(x_source, x_source_lengths, enforce_sorted=self.enforce_sorted_source)
        source_embeddings = source_embeddings.unsqueeze(2) # bsz x hidden x 1
        # We can't sort the target input since it is aligned to source, hence enforce_sorted=False
        target_embeddings = self.encoder(x_target, x_target_lengths, enforce_sorted=False)
        target_embeddings = target_embeddings.mm(self.M).unsqueeze(1) # bsz x 1 x hidden
        concat_embeddings = torch.bmm(target_embeddings, source_embeddings).squeeze()
        return torch.sigmoid(concat_embeddings)
