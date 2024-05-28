import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(x)

class MLPKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, hidden_dim, act):
        super(MLPKANLayer, self).__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.hidden_dim = hidden_dim
        self.proj_1 = nn.Linear(1, hidden_dim, bias=False)
        self.proj_2 = nn.Linear(inputdim * hidden_dim, outdim)
        self.act = act

    def forward(self, x):
        x = self.proj_1(x.unsqueeze(-1)).view(-1, self.inputdim * self.hidden_dim)
        c = self.act(x)
        y = self.proj_2(c)
        return y


class MNISTMLPKAN(nn.Module):
    def __init__(self, act):
        super(MNISTMLPKAN, self).__init__()
        self.mlpkan1 = MLPKANLayer(28 * 28, 128, 28, act)
        self.mlpkan2 = MLPKANLayer(128, 10, 4, act)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.mlpkan1(x)
        x = self.mlpkan2(x)
        return x


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

activations = {
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ELU': nn.ELU(),
    'Cos': Cos(),
    'SiLU': nn.SiLU()
}


# Define the training loop
def train(model, device, train_loader, optimizer, epoch):
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
    model.train()
    for (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()

        progress_bar.set_description(f'Epoch {epoch + 1}/{num_epochs} Running Loss: {running_loss:.6f}')


# Define the evaluation loop
def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


for name, act_fn in activations.items():
    print(f"Testing with {name}")
    model = MNISTMLPKAN(act_fn).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 1
    # Train the model
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch)
    # Evaluate the model
    evaluate(model, device, test_loader)
