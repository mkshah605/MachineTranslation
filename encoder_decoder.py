import torch as t
import torch.nn.functional as F
from torch import nn
from jaxtyping import Float, Int
from textprocessing import TextProcessing

device = t.device("mps")

# Run Text Processing

directory = "/Users/mkshah605/Documents/GitHub/MachineTranslation/corpus_files"

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
        # Step 1: Create the Embedding Matrix
            # input dimensions: tensor[batch_size, sequence_length]
            # output dimensions: tensor[batch_size, sentence_length, embedding_dim]
            # creates the embedding matrix to be of dimensions vocab size x embedding_dim
            # the actual values are random to begin with
            # the embedding matrix is unique (lookup table, key is the index, value is the embedding for that word)
        self.embedding = nn.Embedding(eng_vocab_size, embedding_dim)
        # this just builds the LSTM architechture. Also random floats
        # Step 2: Run the LSTM Model
            # In: input, (h_n, c_n)
            # input dimensions: (for batched) tensor[batch_size, sequence_length, input_size] AKA tensor[batch_size, sentence_length, embedding_dim]
            # h_n input dimensions: tensor[#_layers, hidden_size] of zeros since we don't specify
            # h_c input dimensions: tensor[#_layers, batch_size, hidden_size] of zeros since we don't specify
            # Out: output, (h_n, c_n)
            # output dimensions: (for batched) tensor[batch_size, sequence_length, hidden_size] AKA tensor[batch_size, sentence_length, number_of_LSTMcells]
            # h_n output dimensions: this is the hidden output (short term memory). tensor[#_layers, batch_size, hidden_size]
            # h_c output dimensions: this is the cell output (long term memory). tensor[#_layers, batch_size, hidden_size]
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
        # Step 1: Create the Embedding Matrix
        # input dimensions: tensor[batch_size, sequence_length]
        # output dimensions: tensor[batch_size, sentence_length, embedding_dim]
        # creates the embedding matrix to be of dimensions vocab size x embedding_dim
        # the actual values are random to begin with
        # the embedding matrix is unique (lookup table, key is the index, value is the embedding for that word)
        self.embedding = nn.Embedding(guj_vocab_size, embedding_dim)
        # Step 2: Run the LSTM Model
        # In: input, (h_n, c_n)
        # input dimensions: (for batched) tensor[batch_size, sequence_length, input_size] AKA tensor[batch_size, sentence_length, embedding_dim]
        # h_n input dimensions: tensor[#_layers, hidden_size] of zeros since we don't specify
        # h_c input dimensions: tensor[#_layers, batch_size, hidden_size] of zeros since we don't specify
        # Out: output, (h_n, c_n)
        # output dimensions: (for batched) tensor[batch_size, sequence_length, hidden_size] AKA tensor[batch_size, sentence_length, number_of_LSTMcells]
        # h_n output dimensions: this is the hidden output (short term memory). tensor[#_layers, batch_size, hidden_size]
        # h_c output dimensions: this is the cell output (long term memory). tensor[#_layers, batch_size, hidden_size]
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout_p, batch_first=True)
        # Step 3: Generate the predicted word. This will give us the next word, and will also be used as a hidden layer for the next prediction
        # This is done by performing a linear transormation on 
        # In: any # of dimensions (including none), in_features
        # Out
        self.generator = nn.Linear(in_features=hidden_size, out_features=guj_vocab_size) 

    def to(self, *args, **kwargs):
        on_device = super().to(*args, **kwargs)
        on_device.embedding = on_device.embedding.to("cpu")
        return on_device

    def forward(self, encoder_hidden: t.Tensor):
        # Big picture: 
        # The decoder takes in the hidden layer from the encoder, as well as the previous predicted word
        # For the first round, the previous predicted word is the start token
        # encoder_hidden # of size [batch_size, sequence_length, d_model_encoder] 
        batch_size = encoder_hidden.shape[0] # take the first element (batch size) of the second hidden layer (context vector)
        # why do we do this?? below
        decoder_hidden = encoder_hidden # value from encoder
        decoder_input_int = gu_word2index["<s>"] #  Get the index of the start token. this value is 1
        # goal is to get decoder_input to look like this
        # decoder_input * batch_size [1, 1, 1, 1] (len of batch size)
        # decoder_input should be a tensor of dimension [batch_size]
        decoder_input = t.tensor(t.ones(batch_size)*decoder_input_int).long()
        # forcing the batch size here so it must be 10. This is we why drop 1 seq to make it an eveb multiple of 10
        # can feed in multiple sentences at a time (output is a list of tensors)
        decoder_outputs: list[t.Tensor] = [] # this is a list of 2D tensors. This is why we need stack later to force 3D
        # Our longest gujarati sentence is 46 tokens, so we will predict the next word up tp 50 times. 
        # This limit is in place to prevent our model from running infinitely
        for n in range(50):
            output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            # output is batch size x vocab size
            # but we just need the index of the token (like the original input)
            # this is the predicted token that will be fed into the next step of the model run
            # can do this with the argmax
            # TODO: implement beam search 
            decoder_input = t.argmax(output, dim=-1).detach().to(device) # taking the max prob (explicitly greedy search!) argmax over vocab size!
            # detach here so that we don't compute gradients on this decoder_input
            decoder_outputs.append(output) # we want to save all of the predicted words we get along the way. Why??

        decoder_outputs = t.stack(decoder_outputs, dim=1) #concatenate into tensor of tensors; [batch_size, seq_len, guj_vocab_size], adds another dimension
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1) # softmax all the tensors at one time, over the guj_vocab_size dimension
        return decoder_outputs, decoder_hidden
    
    def forward_step(self, previous_output, hidden):
        """
        Conducts a forward step through one LSTM cell in the decoder.

        Returns:
            output (Tensor[batch_size, guj_vocab_size])
            hidden (Tensor[batch_size, hidden_size])
        """

        # assert previous_output.device == next(self.lstm.parameters).device
        # original_device = previous_output.device
        # breakpoint()
        # output = self.embedding(previous_output.to("cpu")).to(original_device)

        # output = self.embedding(previous_output.to("cpu")).to("mps")
        output = self.embedding(previous_output.to("cpu"))
        output, hidden = self.lstm(output, hidden)
        # output = F.relu(output)
        output = self.generator(output)
        return output, hidden
    


class AttentionDecoder(DecoderLSTM):
    def __init__(self, embedding_dim, guj_vocab_size, hidden_size, num_layers, d_out_kq, d_out_v, dropout_p=0.1):
        super().__init__(embedding_dim, guj_vocab_size, hidden_size, num_layers, dropout_p)

        self.attention = CrossAttention(hidden_size, d_out_kq, d_out_v)
        #calls the parent class to initialize it

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

        print("previous token out shape: ", previous_token_output.shape)
        input = self.embedding(previous_token_output)
        print("encoder hidden shape: ", encoder_out_hc.shape)


        decoder_out, decoder_hidden = self.lstm(input, encoder_out_hc)

        print("encoer output shape: ", encoder_output.shape)
        print("decoder output shape : ", decoder_out.shape)
        print("decoder hn hc: ", decoder_hidden[0].shape, decoder_hidden[1].shape)

        attention_weights, attn_output = self.attention(encoder_output, decoder_out)

        print("attantion output: ", attn_output.shape)
        
        decoder_output = self.generator(attn_output)
        print("decoder_output: ", decoder_output.shape)
        print("decoder_out hidden: ", len(decoder_hidden))
        
        return decoder_output, decoder_hidden, attention_weights
    
    def forward(self, 
                encoder_output: Int[t.Tensor, "batch_size seq_len hidden_size"], encoder_out_hc: Int[t.Tensor, "batch_size d_model_encoder hidden_size"]):
        # Big picture: 
        # The decoder takes in the hidden layer from the encoder, as well as the previous predicted word
        # For the first round, the previous predicted word is the start token
        batch_size = encoder_out_hc.shape[0]

        decoder_input = encoder_output
        decoder_in_hc = encoder_out_hc # value from encoder
        print("decoder hidden shape: ", decoder_in_hc.shape)

        decoder_input_int = gu_word2index["<s>"] #  Get the index of the start token. this value is 1
        decoder_outputs: t.Tensor = t.LongTensor([[decoder_input_int]*batch_size]).T # dim: [batch_size, 1]
        # decoder_outputs is all of our
        # [s s s s s s s s s],
        # [1,2,3,...],
        # [e, e, e, e, e]

        # Our longest gujarati sentence is 46 tokens, so we will predict the next word up tp 50 times. 
        # This limit is in place to prevent our model from running infinitely
        attn_weights = t.Tensor()
        for n in range(50):
            output, decoder_hidden, attention_weights = self.forward_step(decoder_outputs[-1], decoder_input, decoder_in_hc)
            # output is batch size x vocab size
            # but we just need the index of the token (like the original input)
            # this is the predicted token that will be fed into the next step of the model run
            # can do this with the argmax
            # TODO: implement beam search 

            # COMMENTED BELOW
            # attn_weights, new_output = self.attention(encoder_output, output)
            print("output & hidden dimensions", output.shape, decoder_hidden[0].shape, decoder_hidden[1].shape)
            example = t.Tensor(t.ones(4)*3)
            print("example tensor ", output[0:50])
            decoder_input = t.argsort(output, dim=-1, stable=True).detach()
            print("decoder_input", decoder_input[:, :, -1], decoder_input[:, :, -1].shape)
            # argmax returns indexes (argsort apparently does too?)
            decoder_input_2 = t.argmax(output, dim=-1).detach() # taking the max prob (explicitly greedy search!) argmax over vocab size!
            print("decoder_input_2", decoder_input_2.shape)
            # detach here so that we don't compute gradients on this decoder_input
            print("decoder_outputs", decoder_outputs.shape)
            print("output", output.shape)
            decoder_outputs = t.cat([decoder_outputs, output]) # we want to save all of the predicted words we get along the way. Why??
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
        # x_2 is new in cross attention
        # note: if we set x1 = x2, this is the same as self-attention
        # queries (decoder) are from x2 -> this is from gujarati in our case
        # keys and values (encoder) are from x1 -> from english
        # the attention mechanism is evaluating the interaction between two different inputs

        # each context vector is a weighted sum of the values.
        # But unlike self-attention, the values come from the 2nd input (x2) -> which comes from gujarati
        # The weights are based on the interaction between x1 and x2
        print("x1, and x2 shape: ", x_1.shape, x_2.shape)

        queries_1 = x_1 @ self.W_query
        keys_2 = x_2 @ self.W_key
        values_2 = x_2 @ self.W_value

        attn_scores = queries_1 @ keys_2.T 
        attn_weights = t.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights @ values_2

        return attn_weights, context_vec # aim for 128 for size of context vec