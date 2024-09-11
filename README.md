# MachineTranslation

* This project is a Work in Progress *

This repo contains code for an seq2seq Gujarati -> English translation model, written in PyTorch, based on a [2020 paper](https://arxiv.org/pdf/2002.02758). 

## Project Files
#### Git Files
* .gitignore: Python git ignore
* requirements.txt: file containing all necessary packages and imports to run the analysis
#### Directories
* corpus_files: directory containing training sequences and tokens, created by the authors of the original paper
* images: directory with some visualizations of loss during model training
* test_cases: directory for housing tests
#### Files
* workflow.py: file containing a high-level workflow, used to run the  full analysis end-to-end, from data ingenstion to training and beyond.
* training_loop.py: file defining the training and validation steps for the workflow
* textprocessing.py: file containing functions used to parse and clean the sequences and tokens from the input files
* encoder_decoder.py: file containing functions for the encoder and decoders, with commented annotations
* encoder_decoder_clean.py: file containing a clean version (no extra comments or print statements) of encoder_decoder.py


## Model Architechture
![Encoder (2)](https://github.com/user-attachments/assets/ac82b0a4-56d4-4f35-900b-f112346af16a)


## To-Do List
- [x] Encoder & Decoder Code
  - [x] write core encoder and decoder functions
- [x] Functionalize Jupyter Notebook code
- [ ] Run code on GPU
  - [ ] Figure out how to configure to "MPS"
- [ ] Attention Mechanism
  - [x] Build core Attention mechanism
  - [ ] Ensure that decoder accounts for both attention & LSTM
  - [ ] Integrate with workflow.py and training_loop.py
- [ ] Beam Search Algorithm
  - [ ] Understand beam search
  - [ ] Write beam search function in isolation
  - [ ] Integrate beam search with existing codebase
- [ ] Teacher Forcing
  - [ ] Implement teacher forcing during training
- [ ] Test Coverage
  - [ ] write tests for all functions in encoder_decoder.py
