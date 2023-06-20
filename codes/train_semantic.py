import torch
import torch.nn as nn
import torch.optim as optim


def train_semantic_communication_system(channel, images, num_epochs, device, batch_size, lr=0.001):
    # Move the channel to the selected device
    images = images.to(device)
    encoder = channel.encoder.to(device)
    decoder = channel.decoder.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(channel.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        encoded_images = encoder(images)

        # Transmission over AWGN channel
        #noisy_images = add_awgn_noise(encoded_images, snr)

        # Data restoration
        restored_images = decoder(encoded_images)

        # Calculate loss
        loss = criterion(restored_images, images)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return encoder, decoder