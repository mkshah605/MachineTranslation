
import sys, os
sys.path.append("/Users/mkshah605/Documents/GitHub/MachineTranslation")
from encoder_decoder import EncoderLSTM, DecoderLSTM, CrossAttention, AttentionDecoder
import torch as t

batch_size = 30
seq_len = 45
eng_vocab_size = 1500
guj_vocab_size = 2500
enc_input = t.zeros([batch_size, seq_len]).long()

# Create the encoder with all the parameters
encoder = EncoderLSTM(
    embedding_dim=150, 
    eng_vocab_size=eng_vocab_size, 
    hidden_size=128, 
    num_layers= 2, 
    dropout_p=0.1)

# # Create the decoder with all the parameters
decoder = AttentionDecoder(
    embedding_dim=150, 
    guj_vocab_size=guj_vocab_size, 
    hidden_size=128, 
    num_layers= 2, 
    d_out_kq= 64, 
    d_out_v= 128, 
    dropout_p=0.1)
        
def test_attentionforward():
    encoder_out, encoder_out_hidden = encoder(enc_input)
    #print(encoder_out.shape, encoder_out_hidden[0].shape, encoder_out_hidden[1].shape)
    # we only want the data from the context vector
    # here we transpose it to get it in the dim we want
    encoder_out_hc = t.transpose(encoder_out_hidden[1], 0, 1)
    decoder_out, decoder_out_hidden = decoder(encoder_out, encoder_out_hc)
        