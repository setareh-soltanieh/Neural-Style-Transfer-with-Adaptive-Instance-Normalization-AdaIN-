import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from custom_dataset import custom_dataset
from AdaIN_net import AdaIN_net, encoder_decoder
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument("-content_dir", default="Datasets/COCO10k/COCO10k/", help="Directory containing content images")
parser.add_argument("-style_dir", default="Datasets/wikiart10k/wikiart10k/", help="Directory containing style images")
parser.add_argument("-gamma", type=float, default=1.0, help="Style loss weight")
parser.add_argument("-e", type=int, default=20, help="Number of training epochs")
parser.add_argument("-b", type=int, default=32, help="Batch size")
parser.add_argument("-l", default="encoder.pth", help="Path to save encoder model")
parser.add_argument("-s", default="decoder.pth", help="Path to save decoder model")
parser.add_argument("-p", default="decoder.png", help="Path to save output image")
parser.add_argument("-cuda", default="N", choices=["Y", "N"], help="Use GPU (Y) or CPU (N) for training")
parser.add_argument("-save_model_interval", type=int, default=2, help="Intervals for saving models")
parser.add_argument("-save_weights", default="model_10k", help="Path to save model weights")

args = parser.parse_args()

# Use parsed arguments
content_image_dir = args.content_dir
style_image_dir = args.style_dir
gamma = args.gamma
num_epochs = args.e
batch_size = args.b
encoder_model_path = args.l
decoder_model_path = args.s
output_image_path = args.p
use_cuda = args.cuda == "Y"
save_model_interval = args.save_model_interval
save_dir = Path(args.save_weights)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),  # Convert to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

lr = 1e-4

def adjust_learning_rate(optimizer, iteration_count, lr):
    lr_decay = 5e-5
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_conent_dataset = custom_dataset(dir=content_image_dir, transform=transform)
train_style_dataset = custom_dataset(dir=style_image_dir, transform=transform)

train_content_loader = DataLoader(train_conent_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
train_style_loader = DataLoader(train_style_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(torch.cuda.get_device_name(0))  # Print GPU name
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")

# Create an instance of your AdaIN_net
encoder = encoder_decoder.encoder
decoder = encoder_decoder.decoder
encoder.load_state_dict(torch.load(encoder_model_path))
encoder.to(device)
adain_net = AdaIN_net(encoder_decoder.encoder, encoder_decoder.decoder)
adain_net.to(device)

# Define an optimizer
optimizer = optim.Adam(adain_net.decoder.parameters(), lr=lr)

# Creating empty lists for plotting the losses at the end
content_losses = []
style_losses = []
total_losses = []


for i in tqdm(range(num_epochs)):
    adjust_learning_rate(optimizer, iteration_count=i, lr=lr)
    content_images = next(iter(train_content_loader)).to(device)
    style_images = next(iter(train_style_loader)).to(device)
    loss_c, loss_s = adain_net(content_images, style_images)
    loss = loss_c + gamma * loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    content_losses.append(loss_c.item())
    style_losses.append(loss_s.item())
    total_losses.append(loss.item())

    # Log losses to the text file
    print(f'Iteration {i + 1} - Content Loss: {loss_c.item()}, Style Loss: {loss_s.item()}\n, Total Loss: {loss.item()}')

    # Save the model at specified intervals
    if (i + 1) % save_model_interval == 0 or (i + 1) == num_epochs:
        state_dict = adain_net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(device)
        torch.save(state_dict, save_dir / 'decoder_iter_{:d}.pth.tar'.format(i + 1))

# Save the final weights to encoder.pth
torch.save(state_dict, encoder_model_path)

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(content_losses, label="Content Loss")
plt.plot(style_losses, label="Style Loss")
plt.plot(total_losses, label="Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Losses Over Training")
plt.grid(True)

# Save the plot as an image file
plt.savefig("loss_plot_10k.png")
plt.close()