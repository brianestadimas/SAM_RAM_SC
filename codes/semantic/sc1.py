import torch.nn as nn

# Define semantic encoder and decoder models
class SemanticCommunicationChannel(nn.Module):
    def __init__(self):
        super(SemanticCommunicationChannel, self).__init__()
        self.encoder = SemanticEncoder()
        self.decoder = SemanticDecoder()

    def forward(self, x):
        # Encode the input text into a latent representation
        encoded = self.encoder(x)

        # Send the encoded representation through the channel (no-op in this example)
        transmitted = encoded

        # Decode the transmitted representation back into the semantic space
        decoded = self.decoder(transmitted)

        return decoded


class SemanticEncoder(nn.Module):
    def __init__(self):
        super(SemanticEncoder, self).__init__()

        # Define encoder architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional convolution layer and max pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Perform encoding
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        # Additional convolution layer and max pooling
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        return x.sum(dim=0).unsqueeze(0)


class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

        # Define decoder architecture
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(16)

        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm2d(3)

    def forward(self, x):
        # Perform decoding
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        
        x = self.deconv4(x)
        x = self.relu4(x)
        x = self.batchnorm4(x)

        return x.sum(dim=0).unsqueeze(0)