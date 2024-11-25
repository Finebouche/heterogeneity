import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from PIL import Image
import io
from utils import draw_het_network
from reinforce_het import HetNetwork


def run(device):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    batch_size = 128
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size)

    input_dim_nmist = 28 * 28
    num_classes = 10  # MNIST has 10 classes

    het_net = HetNetwork(input_dim=input_dim_nmist, output_dim=num_classes)
    het_net.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(het_net.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        het_net.train()
        total_loss = 0
        total_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # **Flatten the data**
            data = data.view(data.size(0), -1)

            # Forward pass through HetNetwork
            output = het_net(data)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = total_correct / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%')

        # Evaluate on test set
        het_net.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                # **Flatten the data**
                data = data.view(data.size(0), -1)

                output = het_net(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%')

    # Optionally save the trained HetNetwork
    torch.save(het_net.state_dict(), 'het_net_mnist.pth')

    # Optionally visualize the network
    image_data = draw_het_network(het_net, format='png')
    image = Image.open(io.BytesIO(image_data))
    image.show()  # This will display the network visualization

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run(device)