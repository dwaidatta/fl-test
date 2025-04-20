import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import flwr as fl

# 1. Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. Model with dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Data loading with shuffled partitioning
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def partition_dataset(dataset, num_clients):
    indices = np.random.permutation(len(dataset))  # Shuffled, not sorted!
    split_size = len(dataset) // num_clients
    return [Subset(dataset, indices[i*split_size:(i+1)*split_size]) for i in range(num_clients)]

# 4. Training with adjusted hyperparameters
def train(model, trainloader, epochs=3, device=DEVICE):  # Reduced epochs
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

def test(model, testloader, device=DEVICE):
    model.eval()
    model.to(device)
    correct, total, loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return loss / len(testloader), correct / total

# 5. Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=3, device=self.device)  # 3 epochs
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# 6. Simulation with FedProx and 5 clients
def main():
    trainset, testset = load_data()
    num_clients = 5  # Reduced from 10
    partitions = partition_dataset(trainset, num_clients)
    testloader = DataLoader(testset, batch_size=32)

    def client_fn(cid):
        model = Net()
        trainloader = DataLoader(partitions[int(cid)], batch_size=32, shuffle=True)
        return FlowerClient(model, trainloader, testloader, DEVICE).to_client()

    client_resources = {"num_gpus": 0.5 if torch.cuda.is_available() else 0.0}

    # Use FedProx with mu=0.1
    from flwr.server.strategy import FedProx
    strategy = FedProx(proximal_mu=0.1)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
