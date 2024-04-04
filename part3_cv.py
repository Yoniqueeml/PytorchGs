import random
import numpy as np
import torch.cuda
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import Accuracy

train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = train_data.classes

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()

img, label = train_data[0]
print(f'Img: {img}')
print(f'Label: {label}')

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


class MNIST_model(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f'Shape after conv_block_1: {x.shape}') # torch.Size([32, 10, 14, 14])
        x = self.conv_block_2(x)
        # print(f'Shape after conv_block_2: {x.shape}') # torch.Size([32, 10, 7, 7])
        x = self.classifier(x)
        # print(f'Shape after classifier block: {x.shape}') # torch.Size([32, 10])
        return x


dummy_x = torch.randn(size=(1, 28, 28)).unsqueeze(dim=0).to(device)
print(f'Dummy shape: {dummy_x.shape}')
model = MNIST_model(input_shape=1, hidden_units=10, output_shape=10)
try:
    model.load_state_dict(torch.load('MNIST_weights.p'))
except FileNotFoundError:
    model(dummy_x)
    epochs = 5

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_acc = 0
        for batch, (x, y) in enumerate(train_dataloader):
            model.train()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        print(f'\nEpoch: {epoch} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}')

        model.eval()
        with torch.inference_mode():
            test_loss = 0
            test_acc = 0
            for batch, (x, y) in enumerate(test_dataloader):
                y_pred = model(x)
                test_loss += loss_fn(y_pred, y)
                test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            test_loss /= len(test_dataloader)
            train_acc /= len(test_dataloader)
            print(f'Test loss: {train_loss:.3f} | Test acc: {train_acc:.3f}')
    torch.save(model.state_dict(), f='MNIST_weights.p')

model.eval()

num_to_plot = 5

plt.figure(figsize=(10, 8))
for i in range(num_to_plot):
    img = test_data[i][0]
    label = test_data[i][1]

    model_pred_logits = model(img.unsqueeze(dim=0).to(device))
    model_pred_probs = torch.softmax(model_pred_logits, dim=1)
    model_pred_label = torch.argmax(model_pred_probs, dim=1)

    plt.subplot(1, num_to_plot, i + 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Truth: {label} | Pred: {model_pred_label.cpu().item()}")
    plt.axis(False)

plt.tight_layout()
plt.show()

y_preds = []
model.eval()
with torch.inference_mode():
    for batch, (x, y) in tqdm(enumerate(test_dataloader)):
        x, y = x.to(device), y.to(device)
        y_pred_logits = model(x)
        y_pred = torch.softmax(y_pred_logits, dim=1)
        y_label = torch.argmax(y_pred, dim=1)
        y_preds.append(y_label)
    y_preds = torch.cat(y_preds)

confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds, target=test_data.targets)

fix, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()

# Let's try another dataset - FashionMNIST


train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
fashion_classes = train_data.classes

model_Fashion = MNIST_model(input_shape=1,
                            hidden_units=10,
                            output_shape=10).to(device)

try:
    model.load_state_dict(torch.load('MNIST_Fashion_model.p'))
except FileNotFoundError:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_Fashion.parameters(), lr=0.01)
    acc_fn = Accuracy(task='multiclass', num_classes=len(fashion_classes)).to(device)

    epochs = 5
    for epoch in tqdm(range(epochs)):
        train_loss, test_loss_total = 0, 0
        train_acc, test_acc = 0, 0

        model_Fashion.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += acc_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        model_Fashion.eval()
        with torch.inference_mode():
            for batch, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                test_loss_total += loss
                test_acc += acc_fn(y_pred, y)
            test_acc /= len(test_dataloader)
        print(
            f'Epoch: {epoch} '
            f'| Train loss: {train_loss:.2f} '
            f'| Train acc: {train_acc:.2f} '
            f'| Test loss: {test_loss_total:.2f} '
            f'| Test acc: {test_acc:.2f}')
    torch.save(model_Fashion.state_dict(), f='MNIST_Fashion_model.p')

test_preds = []
model_Fashion.eval()
with torch.inference_mode():
    for x_test, y_test in tqdm(test_dataloader):
        y_logits = model_Fashion(x_test.to(device))
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_labels = torch.argmax(y_pred_probs, dim=1)
        test_preds.append(y_pred_labels)
test_preds = torch.cat(test_preds).to(device)

wrong_pred_indexes = np.where(test_preds != test_data.targets)[0]
random_selection = random.sample(list(wrong_pred_indexes), k=9)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(random_selection):
    true_label = fashion_classes[test_data[idx][1]]
    pred_label = fashion_classes[test_preds[idx]]

    plt.subplot(3, 3, i + 1)
    plt.imshow(test_data[idx][0].squeeze(), cmap="gray")
    plt.title(f"True: {true_label} | Pred: {pred_label}", c="r")
    plt.axis(False)
plt.show()
