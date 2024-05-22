import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch import nn
from sklearn.preprocessing import FunctionTransformer
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from DataPreprocessingTools.data_prep_library import rename_columns


"""
Try different lookback values. We have used 30 previous values to make predictions of the current.
Make architecture predict more than one target value. We have created an architecture that only predicts one future target value.
We can create a network that predicts 5 or 10 future target values as well. The data needs to be organized in that way and the output units of the last dense layer should be the same as our selected number of future target values.
Try adding features to data from datetime like a weekday, month-end/month-start, month, AM/PM, etc.
Try different output units for LSTM layers.
Stack more LSTM layers (This can increase training time).
Try adding more dense layers after LSTM layers.
Try different weight initialization methods.
Try learning rate schedulers

Aşağıdaki linkte güzel bir analiz var onu kullan.
https://coderzcolumn.com/tutorials/data-science/how-to-remove-trend-and-seasonality-from-time-series-data-using-python-pandas 
"""

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv(r"C:\Users\pc\DeepLearning\DataSet\household_power_consumption.txt", sep=";", header=None, low_memory=False)

rename_columns(df)

def lstm_variable_fix(dataframe):
    dataframe.columns = ['Date', 'Time', 'Global_active_power',
                         'Global_reactive_power', 'Voltage', 'Global_intensity',
                         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    dataframe = dataframe.drop(0, axis=0)

    dataframe['DateTime'] = pd.to_datetime(dataframe['Date'] + ' ' + dataframe['Time'], format='%d/%m/%Y %H:%M:%S')
    dataframe['Dayofweek_Num'] = dataframe['DateTime'].dt.dayofweek
    dataframe = dataframe.set_index('DateTime')

    dataframe = dataframe.drop(['Date', 'Time'], axis=1)

    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                       'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
        dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
    return dataframe


df = lstm_variable_fix(df)



# df.loc["2010-11"].plot(y="Global_active_power", figsize=(18, 7), color="tomato", grid=True);
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black');


# df.loc["2010-11-26"].plot(y="Global_active_power", figsize=(18, 7), color="black", grid=True)
# plt.grid(which='minor', linestyle=':', linewidth=0.5, color='tomato')
# plt.show()

def lstm_data_scaler_tensor_conversion(dataframe):
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


look_back, num_features, X_train, Y_train, X_test, Y_test = lstm_data_scaler_tensor_conversion(df)


def train_test_data_loader(X_train,Y_train, batch_size):

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in train_loader.dataset.tensors]
    test_loader.dataset.tensors = [tensor.to(torch.device("cuda:0")) for tensor in test_loader.dataset.tensors]

    return train_loader, test_loader


train_loader, test_loader = train_test_data_loader(X_train,Y_train, batch_size=4096) # batchsize = 2048*2

hidden_dim = 128 # Gizli katman boyutunu 512 olarak ayarladık
n_layers = 2     # LSTM katman sayısını 3 olarak ayarladık
num_features = 7 # Giriş özellik sayısını örnek olarak verdik, gerçek değerle değiştirmelisiniz

class LSTMRegressor(nn.Module):
    def __init__(self):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, X_batch):
        hidden, carry = torch.randn(n_layers, len(X_batch), hidden_dim).cuda(), torch.randn(n_layers, len(X_batch), hidden_dim).cuda()
        output, (hidden, carry) = self.lstm(X_batch, (hidden, carry))
        return self.linear(output[:, -1])

# epochs = 10
learning_rate = 0.001

model = LSTMRegressor()
model = model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=True
)

start_time = time.time()
epochs = 20
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

    average_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss}")

    losses.append(average_loss)

print(f"Training time: {time.time() - start_time} seconds")

plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Model Summary:")
print(model)

torch.save(model.state_dict(), 'lstm_model.pth')
model.load_state_dict(torch.load('lstm_model.pth'))
print(model.eval())

X_test_sample = X_test[:10000]  # yaklaşık 17 saatlik veri örneğinin alınması
Y_test_sample = Y_test[:10000]

model.eval()
with torch.no_grad():
    Y_pred_sample = model(X_test_sample)

Y_test_sample = Y_test_sample.cpu().numpy()
Y_pred_sample = Y_pred_sample.cpu().numpy()

datetime_index = df.index[1452682:1452682 + 10000]

plt.figure(figsize=(18, 7))
plt.plot(datetime_index, Y_test_sample, label='Gerçek Değerler', color='blue')
plt.plot(datetime_index, Y_pred_sample, label='Tahmin Değerleri', color='red')
plt.xlabel('Tarih')
plt.ylabel('Global Active Power')
plt.title('Gerçek ve Tahmin Edilen Global Active Power Değerleri')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('LSTM_Truth_vs_Prediction.png')
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
plt.savefig('LSTM_Time_Series_Feature_Importance.png')


