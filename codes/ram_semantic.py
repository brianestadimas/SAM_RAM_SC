from semantic.sc_linear import SemanticCommunicationChannel
import torch
import torchtext
from torchtext.vocab import GloVe
import torchtext.vocab as vocab

# Pass the input through the channel, get from ram_inference.py
input_text = "arm | beak | bird | catch | cockatoo | hand | person | man | parrot | perch | pet | white | yellow"

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
embedding_dim = 300
vectors = torchtext.vocab.build_vocab_from_iterator(tokenizer(input_text))

vocab = vocab.GloVe(name='6B', dim=embedding_dim, cache='./.vector_cache')

# Tokenize input text
tag_list = tokenizer(input_text)

# Get the indices for the tokens in the GloVe vocabulary
input_indices = [vocab.stoi[token] for token in tag_list]
input_tensor = torch.tensor(input_indices)

# Semantic Channel
channel = SemanticCommunicationChannel()

output_tensor = channel(input_tensor.unsqueeze(0))  # Add an extra dimension for batch size

# Decode the output tensor
output_indices = output_tensor.squeeze(0).tolist()
output_tags = [vocab.itos[index] for index in output_indices]

print(output_tags)