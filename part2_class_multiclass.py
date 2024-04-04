import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import Accuracy

from helper_functions import plot_decision_boundary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_SAMPLES = 1000
RANDOM_SEED = 42

X, Y = make_moons(n_samples=NUM_SAMPLES, noise=0.05, random_state=RANDOM_SEED)

data_df = pd.DataFrame({'X0': X[:, 0], 'X1': X[:, 1], 'Y': Y})
colors = ['black' if label == 0 else 'red' for label in Y]
plt.scatter(X[:, 0], X[:, 1], c=colors)
# plt.show()

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)


class MoonModelV0(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=10):
        super().__init__()

        self.layer1 = nn.Linear(in_features=in_features,
                                out_features=hidden_units)
        self.layer2 = nn.Linear(in_features=hidden_units,
                                out_features=hidden_units)
        self.layer3 = nn.Linear(in_features=hidden_units,
                                out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


model_0 = MoonModelV0(in_features=2, out_features=1).to(device)
print(model_0)
print(model_0.state_dict())
loss_fn = nn.BCEWithLogitsLoss()  # sigmoid layer built-in
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)


print(f'Logits: {model_0(x_train.to(device)[:10]).squeeze()}')
print(f'Pred probs: {torch.sigmoid(model_0(x_train.to(device)[:10]).squeeze())}')
print(f'Pred labels: {torch.round(torch.sigmoid(model_0(x_train.to(device)[:10]).squeeze()))}')

acc_fn = Accuracy(task='multiclass', num_classes=2).to(device)
torch.manual_seed(RANDOM_SEED)
epochs = 6000
x_train.to(device)
y_train.to(device)
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(x_train).squeeze()
    y_preds_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(y_preds_probs)

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train.int())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test.int())

    if epoch % 100 == 0:
        print(
            f'Epoch: {epoch} | Loss: {loss:.2f} | Acc: {acc:.2f} | Test loss: {test_loss:.2f} | Test acc: {test_acc:.2f}')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, x_test, y_test)
plt.show()


def tanh(x):
    # Source - https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


# __________________________________SPIRAL_________________________
# __________________________________SPIRAL_________________________
# __________________________________SPIRAL_________________________


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
Y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    Y[ix] = j
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40)
plt.show()

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.LongTensor)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)
acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"


class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))


model_1 = SpiralModel().to(device)

print(model_1.state_dict())

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
print("Logits:")
print(model_1(x_train)[:10])
print("Pred probs:")
print(torch.softmax(model_1(x_train)[:10], dim=1))
print("Pred labels:")
print(torch.softmax(model_1(x_train)[:10], dim=1).argmax(dim=1))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.02)
# Build a training loop for the model
epochs = 1000

for epoch in range(epochs):
    model_1.train()
    y_logits = model_1(x_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(x_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {acc:.2f} | Test loss: {test_loss:.2f} Test acc: {test_acc:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, x_test, y_test)
plt.show()