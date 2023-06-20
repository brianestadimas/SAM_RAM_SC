import torch
import torchtext.vocab as vocab
from tokenizers import ByteLevelBPETokenizer
from semantic.sc_linear import SemanticCommunicationChannel

# Pass the input through the channel, get from ram_inference.py
input_text = "arm | beak | bird | catch | cockatoo | hand | person | man | parrot | perch | pet | white | yellow"

# Define and train the BPE tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train([input_text])

embedding_dim = 300

# Create the vocabulary using BPE
vocab = vocab.build_vocab_from_iterator([tokenizer.get_vocab()])
vocab.set_vectors(vocab.stoi, torch.randn(len(vocab), embedding_dim), embedding_dim)

# Tokenize input text
tag_list = tokenizer.encode(input_text).tokens

# Get the indices for the tokens in the BPE vocabulary
input_indices = [vocab[token] for token in tag_list]
input_tensor = torch.tensor(input_indices)

# Semantic Channel
channel = SemanticCommunicationChannel()

output_tensor = channel(input_tensor.unsqueeze(0))  # Add an extra dimension for batch size

# Decode the output tensor
output_indices = output_tensor.squeeze(0).tolist()
output_tags = [vocab.itos[index] for index in output_indices]

print(output_tags)
