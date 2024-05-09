import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from DataPreprocessingTools.data_prep_library import rename_columns, variable_fix, feature_importance, quick_look

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv(r"/DataSet/household_power_consumption.txt", sep=";", header=None, low_memory=False)

quick_look(df)
rename_columns(df)
df = variable_fix(df)

def data_prep(data):
    x = data.drop("Global_active_power", axis=1).values
    y = data["Global_active_power"].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    input_dim = X_train[1]

    X_train = torch.FloatTensor(X_train).cuda()
    X_test = torch.FloatTensor(X_test).cuda()

    y_train = (torch.FloatTensor(y_train).reshape(-1, 1)).cuda()
    y_test = (torch.FloatTensor(y_test).reshape(-1, 1)).cuda()


    return X_train, X_test, y_train, y_test, input_dim

X_train, X_test, y_train, y_test, input_dim = data_prep(df)

batch_size = 4096
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in train_loader.dataset.tensors]
test_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in test_loader.dataset.tensors]

# print(train_loader.dataset.tensors[0].device)
# print(test_loader.dataset.tensors[0].device)
class linearRegression(nn.Module):
    def __init__(self, input_dim):
        super(linearRegression, self).__init__()

        # Define your layers here using input_dim
        self.fc1 = nn.Linear(input_dim, 7)
        init.xavier_uniform_(self.fc1.weight)
        self.batch_norm1 = nn.BatchNorm1d(7)

        self.fc2 = nn.Linear(7, 4)
        init.xavier_uniform_(self.fc1.weight)
        self.batch_norm1 = nn.BatchNorm1d(4)

        self.fc3 = nn.Linear(4, 1)
        init.xavier_uniform_(self.fc1.weight)
        self.batch_norm1 = nn.BatchNorm1d(1)


    def forward(self, x):
        # Define forward pass here
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return x

torch.manual_seed(42)
model = linearRegression(input_dim=8)
model = model.cuda()
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=True
)


start_time = time.time()
epochs = 10
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = loss_function(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

print(f"Training time: {time.time() - start_time} seconds")

plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Model Summary:")
print(model)

torch.save(model.state_dict(), 'linear_regression_model.pth')
model.load_state_dict(torch.load('linear_regression_model.pth'))
print(model.eval())











