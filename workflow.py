import torch as t

from textprocessing import TextProcessing
from encoder_decoder import EncoderLSTM, DecoderLSTM, AttentionDecoder, CrossAttention
from training_loop import TrainingLoop
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split

device = t.device("mps")

# Run Text Processing
directory = "/Users/mkshah605/Documents/GitHub/MachineTranslation/corpus_files"

en_corpus_file = "train_en.txt"
gu_corpus_file = "train_gu.txt"

en_tokens_file = "vocab_en.txt"
gu_tokens_file = "vocab_gu.txt"

gu_sentences, gu_words, gu_word2index, gu_index2word, guj_seq_collection = TextProcessing.run_text_processing(TextProcessing(), directory=directory, corpus_file=gu_corpus_file, vocab_file=gu_tokens_file, max_sent_len=50)
en_sentences, en_words, en_word2index, en_index2word, eng_seq_collection = TextProcessing.run_text_processing(TextProcessing(), directory=directory, corpus_file=en_corpus_file, vocab_file=en_tokens_file, max_sent_len=55)

#######################################################################################

batch_size = 50 # can experiment with larger batch sizes
eng_vocab_size = len(en_words)
guj_vocab_size = len(gu_words)


# Create the encoder with all the parameters
encoder = EncoderLSTM(
    embedding_dim=150, 
    eng_vocab_size=eng_vocab_size, 
    hidden_size=128, 
    num_layers= 2, 
    dropout_p=0.1).to(device)


# Create the decoder with all the parameters
decoder = DecoderLSTM(
    embedding_dim=150, 
    guj_vocab_size=guj_vocab_size, 
    hidden_size=128, 
    num_layers= 2, 
    dropout_p=0.1).to(device)


# # Create the Attention decoder with all the parameters
decoder = AttentionDecoder(
    embedding_dim=150, 
    guj_vocab_size=guj_vocab_size, 
    hidden_size=128, 
    num_layers= 2, 
    d_out_kq= 64, 
    d_out_v= 128, 
    dropout_p=0.1)

# Create the attention layer with all the parameters
t.manual_seed(123)
# d_in, d_out_kq, d_out_v = 150, 64, 64
# crossattn = CrossAttention(d_in, d_out_kq, d_out_v)

# first_input = embedded_sentence
# second_input = t.rand(8, d_in)

# print("First input shape:", first_input.shape)
# print("Second input shape:", second_input.shape)


#############################################################################


# Split Dataset
# Note: we index all sentences except the last sentence to bring the total number of sequences for each language toan even number: 6500
# We do this because our decoder can only handle full batches
data = TensorDataset(t.LongTensor(eng_seq_collection[:-1]).to(device),
                               t.LongTensor(guj_seq_collection[:-1]).to(device))


train_dataset, val_dataset = random_split(data, [0.7, 0.3])


# Set up Training & Validation DataLoaders
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

val_sampler = RandomSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

# Run Training Loop
TrainingLoop.train(TrainingLoop(), train_dataloader, val_dataloader, encoder, decoder, n_epochs=1)


