import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from sklearn.metrics import mean_squared_error
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from DataPreprocessingTools.data_prep_library import rename_columns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv(r"C:\Users\pc\DeepLearning\DataSet\household_power_consumption.txt", sep=";", header=None, low_memory=False)

def rnn_variable_fix(df):
    rename_columns(df)
    df = df.drop(0, axis=0)
    df.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                  'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d/%m/%Y')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df['Dayofweek_Num'] = df['datetime'].dt.dayofweek
    df = df.set_index('datetime')
    df = df.drop(['Date', 'Time'], axis=1)
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())

    return df
df = rnn_variable_fix(df)
def RNN_data_scaler_tensor_conversion(dataframe):
    features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Dayofweek_Num']
    target = ['Global_active_power']

    X = dataframe[features].values
    y = dataframe[target].values

    num_features = X.shape[1]

    look_back = 60 #1 saatlik tahmin için alınacak örnek sayısı veri dakkikalık olarak tutulmuş.

    X_organized, Y_organized = [], []
    for i in range(0, X.shape[0] - look_back, 1):
        X_organized.append(X[i:i + look_back])
        Y_organized.append(y[i + look_back])

    X_organized, Y_organized = np.array(X_organized), np.array(Y_organized)
    X_organized, Y_organized = torch.tensor(X_organized, dtype=torch.float32), torch.tensor(Y_organized, dtype=torch.float32)
    X_train, Y_train, X_test, Y_test = X_organized[:1452682], Y_organized[:1452682], X_organized[1452682:], Y_organized[1452682:]

    std_scaler = StandardScaler()
    Y_train = std_scaler.fit_transform(Y_train)
    Y_test = std_scaler.fit_transform(Y_test)

    X_train = torch.FloatTensor(X_train).cuda()
    X_test = torch.FloatTensor(X_test).cuda()

    Y_train = (torch.FloatTensor(Y_train).reshape(-1, 1)).cuda()
    Y_test = (torch.FloatTensor(Y_test).reshape(-1, 1)).cuda()

    return look_back, num_features, X_train, Y_train, X_test, Y_test


look_back, num_features, X_train, Y_train, X_test, Y_test = RNN_data_scaler_tensor_conversion(df)


def train_test_data_loader(X_train,Y_train, batch_size):

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in train_loader.dataset.tensors]
    test_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in test_loader.dataset.tensors]

    return train_loader, test_loader


train_loader, test_loader = train_test_data_loader(X_train, Y_train, batch_size=4096)

class RNN_Time_Series_Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_ratio):
        super(RNN_Time_Series_Regression, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_ratio)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, updated_data):
        h_0 = torch.zeros(self.layer_dim, updated_data.size(0), self.hidden_dim, device=updated_data.device).requires_grad_()
        out, h_0 = self.rnn(updated_data, h_0.detach())
        out = F.relu(out)
        out = out[:, -1, :]  # Son zaman adımının çıktısını al
        out = self.fc1(out)
        return out

input_dim = num_features
hidden_dim = 2 ** 6
layer_dim = 5
output_dim = 1
dropout_ratio = 0.3

model = RNN_Time_Series_Regression(input_dim, hidden_dim, layer_dim, output_dim, dropout_ratio).cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=True)

start_time = time.time()
epochs = 10
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

torch.save(model.state_dict(), 'RNN_Time_Series_Regression.pth')
model.load_state_dict(torch.load('RNN_Time_Series_Regression.pth'))
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

Y_test_sample = Y_test_sample.cpu().numpy()
Y_pred_sample = Y_pred_sample.cpu().numpy()

datetime_index = df.index[1452682:1452682 + 10000]

# Gerçek ve tahmin edilen değerleri çizdir
plt.figure(figsize=(18, 7))
plt.plot(datetime_index, Y_test_sample, label='Gerçek Değerler', color='blue')
plt.plot(datetime_index, Y_pred_sample, label='Tahmin Değerleri', color='red')
plt.xlabel('Tarih')
plt.ylabel('Global Active Power')
plt.title('RNN İle Zaman Serisi Regresyon Modeli Gerçek ve Tahmin Edilen Global Active Power Değerleri')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('RNN_Time_Series_Regression_Reg_prediction_vs_Test.png')

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
plt.savefig('RNN_Time_Series_Feature_Importance.png')













