import torch as t
import torch.nn.functional as F
from torch import nn
from jaxtyping import Float, Int
from textprocessing import TextProcessing

device = t.device("mps")

# Run Text Processing

directory = "/Users/mkshah605/Documents/GitHub/RC/NMT_IndicLang/corpus_files"

en_corpus_file = "train_en.txt"
gu_corpus_file = "train_gu.txt"

en_tokens_file = "vocab_en.txt"
gu_tokens_file = "vocab_gu.txt"
gu_sentences, gu_words, gu_word2index, gu_index2word, guj_seq_collection = TextProcessing.run_text_processing(TextProcessing(), directory=directory, corpus_file=gu_corpus_file, vocab_file=gu_tokens_file, max_sent_len=50)


class EncoderLSTM(nn.Module):
    def __init__(self, embedding_dim, eng_vocab_size, hidden_size, num_layers, dropout_p=0.1):
        """
        embedding_dim: for each word in vocab_size, this is the number of floats representing each word
        vocab_size: the size of the original english vocab len(en_words)
        hidden_size: the number of LSTM cells in each layer
        num_layers: the number of LSTM layers in the model
        dropout_p: 
        """
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(eng_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout_p, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderLSTM(nn.Module):
    def __init__(self, embedding_dim, guj_vocab_size, hidden_size, num_layers, dropout_p=0.1):
        """
        embedding_dim: for each word in guj_vocab_size, this is the number of floats representing each word
        guj_vocab_size: the size of the original gujarati vocab len(gu_words)
        hidden_size: the number of LSTM cells in each layer
        num_layers: the number of LSTM layers in the model
        dropout_p: 
        """
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(guj_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout_p, batch_first=True)
        self.generator = nn.Linear(in_features=hidden_size, out_features=guj_vocab_size) 

    def to(self, *args, **kwargs):
        on_device = super().to(*args, **kwargs)
        on_device.embedding = on_device.embedding.to("cpu")
        return on_device

    def forward(self, encoder_hidden: t.Tensor):
        batch_size = encoder_hidden.shape[0] 
        decoder_hidden = encoder_hidden # value from encoder
        decoder_input_int = gu_word2index["<s>"] #  Get the index of the start token. this value is 1
        decoder_input = t.tensor(t.ones(batch_size)*decoder_input_int).long()
        decoder_outputs: list[t.Tensor] = [] 
        for n in range(50):
            output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_input = t.argmax(output, dim=-1).detach().to(device)
            decoder_outputs.append(output)

        decoder_outputs = t.stack(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden
    
    def forward_step(self, previous_output, hidden):
        """
        Conducts a forward step through one LSTM cell in the decoder.

        Returns:
            output (Tensor[batch_size, guj_vocab_size])
            hidden (Tensor[batch_size, hidden_size])
        """
        output = self.embedding(previous_output.to("cpu"))
        output, hidden = self.lstm(output, hidden)
        output = self.generator(output)
        return output, hidden
    


class AttentionDecoder(DecoderLSTM):
    def __init__(self, embedding_dim, guj_vocab_size, hidden_size, num_layers, d_out_kq, d_out_v, dropout_p=0.1):
        super().__init__(embedding_dim, guj_vocab_size, hidden_size, num_layers, dropout_p)

        self.attention = CrossAttention(hidden_size, d_out_kq, d_out_v)

    def forward_step(self, 
                     previous_token_output: Int[t.Tensor, "batch_size seq_len"], 
                     encoder_output: Int[t.Tensor, "batch_size seq_len hidden_size"], 
                     encoder_out_hc: Int[t.Tensor, "batch_size d_model_encoder hidden_size"]):
        # seqlen = 1 just the curent token 
        """
        Conducts a forward step through one LSTM cell in the decoder.

        Returns:
            decoder_output (Tensor[batch_size, guj_vocab_size])
            decoder_hidden (Tensor[batch_size, hidden_size])
        """
        decoder_output: Float[t.Tensor, "batch_size 1 hidden_size"]

        input = self.embedding(previous_token_output)

        decoder_out, decoder_hidden = self.lstm(input, encoder_out_hc)
        attention_weights, attn_output = self.attention(encoder_output, decoder_out)        
        decoder_output = self.generator(attn_output)
        
        return decoder_output, decoder_hidden, attention_weights
    
    def forward(self, 
                encoder_output: Int[t.Tensor, "batch_size seq_len hidden_size"], 
                encoder_out_hc: Int[t.Tensor, "batch_size d_model_encoder hidden_size"]):

        batch_size = encoder_out_hc.shape[0]

        decoder_input = encoder_output
        decoder_in_hc = encoder_out_hc # value from encoder

        decoder_input_int = gu_word2index["<s>"] #  Get the index of the start token. this value is 1
        decoder_outputs: t.Tensor = t.LongTensor([[decoder_input_int]*batch_size]).T # dim: [batch_size, 1]

        attn_weights = t.Tensor()
        for n in range(50):
            output, decoder_hidden, attention_weights = self.forward_step(decoder_outputs[:, -1], decoder_input, decoder_in_hc)
            decoder_input = t.argmax(output, dim=-1).detach() # taking the max prob (explicitly greedy search!) argmax over vocab size!
            decoder_outputs = t.cat([decoder_outputs, output], dim=1) # we want to save all of the predicted words we get along the way. Why??
            attn_weights = t.cat([attn_weights, attention_weights])
            
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1) # softmax all the tensors at one time, over the guj_vocab_size dimension
        return decoder_outputs, decoder_hidden
    
class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        # d_in: embedding size
        # d_out_qk:
        # d_out_v: 

        # note: keys and values usually come from encoder and queries come from decoder
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(t.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(t.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(t.rand(d_in, d_out_v))

    def forward(self, 
                x_1: Int[t.Tensor, "batch_size seq_len hidden_size"], 
                x_2):
        
        if len(x_2.shape) == 2:
            x_2 = x_2.unsqueeze(1) # make the second tensor 3D from 2D

        queries_1 = x_1 @ self.W_query
        keys_2 = x_2 @ self.W_key
        values_2 = x_2 @ self.W_value

        # attn_scores = queries_1 @ keys_2.T
        attn_scores: Float[t.Tensor, "batch_size seq_len 1"] = t.bmm(queries_1, keys_2.transpose(1, 2))
        attn_weights = t.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights @ values_2

        return attn_weights, context_vec # aim for 128 for size of context vec