import torch
import matplotlib.pyplot as plt
from torch import nn

weight = 0.3
bias = 0.9

torch.manual_seed(552)

X = torch.rand(100, dtype=torch.float32)
Y = weight * X + bias
train_split = int(0.8 * len(X))
x_train, y_train = X[train_split:], Y[train_split:]
x_test, y_test = X[:train_split], Y[:train_split]

plt.scatter(x_train, y_train, c='g', s=4, label='train data')
plt.scatter(x_test, y_test, c='r', s=4, label='test data')
plt.legend()
plt.show()


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # version 1
        self.weights = nn.Parameter(torch.rand(1, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float32), requires_grad=True)
        # version 2
        # self.lin = nn.Linear(input_size, output_size) input/output should be in construct

    def forward(self, x):
        return self.weights * x + self.bias


model = CustomModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
epochs = 4000

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    if epoch % 20 == 0:
        with torch.inference_mode():
            test_pred = model(x_test)
            test_loss = loss_fn(y_test, test_pred)
        print(f'Epoch: {epoch} | Train Loss: {loss} | Test Loss {test_loss}')

print('Test trained model')
with torch.inference_mode():
    y_pred = model(x_test)
    plt.scatter(x_test, y_test, c='g', s=3, label='test')
    plt.scatter(x_test, y_pred, c='r', s=2, label='pred')
    plt.legend()
    plt.show()

print(f'params: {model.state_dict()}')
# save model (only params)
torch.save(obj=model.state_dict(), f='trained_model.p')

# load model
model2 = CustomModel()
model2.load_state_dict(torch.load('trained_model.p'))
print(f'params: {model2.state_dict()}')
