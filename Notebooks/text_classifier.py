import torch
import torch.nn as nn
from torch.utils.data import DataLoader

data = [[[5, 10], 0], [7, 18], [10, 8], [15, 1], [20, 33], [23, 19]]
labels = [0, 0, +1, +1, 0, 0]

data = [[[5, 10], 0], [[7, 18], 0], [[10, 8], 1], [[15, 1], 1], [[20, 33], 0], [[23, 19], 0]]

train_dataloader = DataLoader(data, batch_size=1)
test_dataloader = DataLoader(data, batch_size=1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
        )

    def forward(self, x): # Forward Propagation
        x = self.flatten(x) # 차원 맞추기, 여기서는 의미 없음
        logits = self.linear_relu_stack(x)
        return logits

device = "cpu"

model = NeuralNetwork().to(device)

print(f"model: {model}")

X = torch.rand(1, 2, device=device)
print(f"X: {X}")
logits = model(X)
print(f"logits: {logits}")
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print(f"Size of X: {X.size()}")
print(f"Predicted class: {y_pred}")
print(f"Predicted class size: {y_pred.size()}")
print(f"Predicted value: {y_pred[0]}")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


learning_rate = 0.001
batch_size = 1
epochs = 3

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")