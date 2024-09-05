from itertools import repeat

class TextProcessing:
    def __init__(self):
        """
        """

    def load_sentencefiles(self, directory, corpus_file):
        """
        Loads sentence files data. Separates by newline character.

        directory: the path of the directory where the corpus/sentence files are located.  
        corpus_file: the filename + extension of the file containing the sentences.
        """
        file = open(directory + "/" + corpus_file, "r", encoding="utf-8")
        sentences = file.read()
        file.close()
        list_of_sentences = sentences.split('\n')
        return list_of_sentences
    

    def load_vocabfiles(self, directory, vocab_file):
        """
        Loads vocab files data. Separates by newline character, removes zero-byte characters, and adds a pad token to the vocab list.
        Returns a list of words/tokens.

        directory: the path of the directory where the corpus/sentence files are located.  
        vocab_file: the filename + extension of the file containing the sentences.
        """
        file = open(directory + "/" + vocab_file, "r")
        words = file.read()
        file.close()
        words = words.split('\n')
        # get rid of the zero-byte characters
        word_list = [word.replace("\u200b","") for word in words]
        # add pad token to vocab list
        word_list.append("</pad>")
        return word_list


    def build_dicts_and_tokenize(self, word_list):
        """
        Tokenizes given word_list into indexes. 
        Returns 2 dictionaries: word2index and index2word.

        word_list: a list of words. Ideally, the output of load_vocabfiles.
        """
        word2index = {}
        index2word = {}
        for i, word in enumerate(word_list):
            word2index[word] = i
            index2word[i] = word
        return word2index, index2word


    def build_seq_collection(self, max_sent_len, list_of_sentences, word2index):
        """
        Builds a list of tokenized and indexed sequences from the list_of_sentences. 
        Each sequence starts with a start token, ends with a stop token, and is then padded to meet the max_sent_len so they are all of the same size.

        max_sent_len: The final sequence length (with padding). This must be greater than the number of sentences for each language.  
        list_of_sentences: A list of sentences. Ideally, the output of load_sentencefiles.  
        word2index: Dictionary word2index from build_dicts_and_tokenize
        """
        seq_collection = []
        for sentence in list_of_sentences:
            seq_sentence = []
            seq_sentence.append(word2index['<s>'])

            for word in sentence.strip().split(' '):
                word = word.replace("\u200b","")
                seq_sentence.append(word2index[word])

            seq_sentence.append(word2index['</s>'])
            difference = max_sent_len - len(seq_sentence)

            if difference > 0:
                seq_sentence.extend(repeat(word2index["</pad>"], difference))

            seq_collection.append(seq_sentence)

        return seq_collection
    
    def run_text_processing(self, directory, corpus_file, vocab_file, max_sent_len):
        """
        Runs all the text processing steps using the helper functions in this class:
        - Loads Sentence Files
        - Loads Vocab Files
        - Builds Dicts and Tokenizes
        - Builds Sequence Collection  

        directory: the path of the directory where the corpus/sentence files are located.  
        corpus_file: the filename + extension of the file containing the sentences.  
        vocab_file: the filename + extension of the file containing the sentences.  
        max_sent_len: The final sequence length (with padding). This must be greater than the number of sentences for each language.  
        """
        sentences = self.load_sentencefiles(directory, corpus_file)
        word_list =  self.load_vocabfiles(directory, vocab_file)
        word2index, index2word = self.build_dicts_and_tokenize(word_list)
        seq_collection = self.build_seq_collection(max_sent_len=max_sent_len, list_of_sentences=sentences, word2index=word2index)
        return sentences, word_list, word2index, index2word, seq_collection









