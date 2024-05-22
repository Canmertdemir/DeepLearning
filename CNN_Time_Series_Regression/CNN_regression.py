import time

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from DataPreprocessingTools.data_prep_library import rename_columns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv(r"C:\Users\pc\DeepLearning\DataSet\household_power_consumption.txt", sep=";", header=None, low_memory=False)

def tm_variable_fix(df):
    rename_columns(df)
    df = df.drop(0, axis=0)
    df.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                  'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d/%m/%Y')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('datetime')

    df = df.drop(['Date', 'Time'], axis=1)

    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())

    return df


df = tm_variable_fix(df)


def data_scaler_tensor_conversion(dataframe):
    features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    target = ['Global_active_power']

    X = dataframe[features].values
    y = dataframe[target].values

    num_features = X.shape[1]


    X_train, X_test = X[:1452682], X[1452682:]
    Y_train, Y_test = y[:1452682], y[1452682:]

    X_train = torch.FloatTensor(X_train).unsqueeze(2).cuda()
    X_test = torch.FloatTensor(X_test).unsqueeze(2).cuda()

    Y_train = torch.FloatTensor(Y_train).cuda()
    Y_test = torch.FloatTensor(Y_test).cuda()

    return num_features, X_train, Y_train, X_test, Y_test


num_features, X_train, Y_train, X_test, Y_test = data_scaler_tensor_conversion(df)


def train_test_data_loader(X_train, Y_train, X_test, Y_test, batch_size):
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in train_loader.dataset.tensors]
    test_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in test_loader.dataset.tensors]

    return train_loader, test_loader


train_loader, test_loader = train_test_data_loader(X_train, Y_train, X_test, Y_test, batch_size=2048)


# Örnek verileri oluşturma
seq_length = 60  # Her bir zaman serisi örneğinin uzunluğu
input_channels = X_train.shape[1]  # Giriş kanal sayısı
# CNN modelini oluşturma
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 50)  # Lineer katmanın giriş boyutunu 64 olarak düzeltin
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNNModel()
model = CNNModel().cuda()

# Loss fonksiyonu ve optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
epochs = 20
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss}")

    losses.append(average_loss)

print(f"Training time: {time.time() - start_time} seconds")

plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()

plt.savefig('training_loss_plot.png')

print("Model Summary:")
print(model)

torch.save(model.state_dict(), 'CNN_regression_model.pth')
model.load_state_dict(torch.load('CNN_regression_model.pth'))
print(model.eval())

model.eval()
predictions = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        y_pred = model(batch_X)
        predictions.append(y_pred.cpu().numpy())


predictions = np.concatenate(predictions)


y_test_numpy = Y_test.cpu().numpy()
y_test_sample = y_test_numpy[:1000]
predictions_sample = predictions[:1000]

X_test_sample = X_test[:10000]  # yaklaşık 17 saatlik veri örneğinin alınması
Y_test_sample = Y_test[:10000]


model.eval()
with torch.no_grad():
    Y_pred_sample = model(X_test_sample)

# Tahmin ve gerçek değerleri CPU'ya taşı ve numpy array'ine dönüştür
Y_test_sample = Y_test_sample.cpu().numpy()
Y_pred_sample = Y_pred_sample.cpu().numpy()

# Test verilerini tarihlere göre al
datetime_index = df.index[1452682:1452682 + 10000]

# Gerçek ve tahmin edilen değerleri çizdir
plt.figure(figsize=(18, 7))
plt.plot(datetime_index, Y_test_sample, label='Gerçek Değerler', color='blue')
plt.plot(datetime_index, Y_pred_sample, label='Tahmin Değerleri', color='red')
plt.xlabel('Tarih')
plt.ylabel('Global Active Power')
plt.title('CNN İle Zaman Serisi Regresyon Modeli Gerçek ve Tahmin Edilen Global Active Power Değerleri')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('CNN_Time_Series_Reg_prediction_vs_Test.png')

def permutation_importance(model, X_test, y_test, num_features, metric=mean_squared_error):
    baseline_score = metric(y_test.cpu().numpy(), model(X_test).cpu().numpy())
    importances = []

    for i in range(num_features):
        X_test_permuted = X_test.clone()
        X_test_permuted[:, :, i] = X_test_permuted[:, torch.randperm(X_test_permuted.size(1)), i]
        permuted_score = metric(y_test.cpu().numpy(), model(X_test_permuted).cpu().numpy())
        importances.append(baseline_score - permuted_score)

    return importances

model.eval()
with torch.no_grad():
    importances = permutation_importance(model, X_test[:10000], Y_test[:10000], num_features)

feature_names = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Dayofweek_Num']
plt.figure(figsize=(18, 7))
plt.barh(feature_names, importances)
plt.xlabel('Performanslar')
plt.title('Özelik Önemleri')
plt.grid(True)
plt.show()
plt.savefig('CNN_Time_Series_Feature_Importance.png')

