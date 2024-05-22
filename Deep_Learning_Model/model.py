import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from DataPreprocessingTools.data_prep_library import rename_columns,quick_look

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv(r"C:\Users\pc\DeepLearning\DataSet\household_power_consumption.txt", sep=";", header=None, low_memory=False)

quick_look(df)
rename_columns(df)

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
def feature_enginerring(dataframe):

    print(dataframe['Date'].head())

    dataframe['Date'] = pd.to_datetime(dataframe['Date'], dayfirst=True, format='%d/%m/%Y', errors='coerce')
    if dataframe['Date'].isnull().any():
        print("Warning: Some dates were not parsed correctly. Check the format of the 'Date' column.")

    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Day'] = dataframe['Date'].dt.day

    dataframe['Time'] = pd.to_datetime(dataframe['Time'], format='%H:%M:%S', errors='coerce').dt.time
    if dataframe['Time'].isnull().any():
        print("Warning: Some times were not parsed correctly. Check the format of the 'Time' column.")

    dataframe['Hour'] = dataframe['Time'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
    dataframe['Minute'] = dataframe['Time'].apply(lambda x: x.minute if pd.notnull(x) else np.nan)
    dataframe['Second'] = dataframe['Time'].apply(lambda x: x.second if pd.notnull(x) else np.nan)

    dataframe["Year_Sin"] = sin_transformer(1).fit_transform(dataframe[["Year"]])
    dataframe["Year_Cos"] = cos_transformer(1).fit_transform(dataframe[["Year"]])

    dataframe["Month_Sin"] = sin_transformer(12).fit_transform(dataframe[["Month"]])
    dataframe["Month_Cos"] = cos_transformer(12).fit_transform(dataframe[["Month"]])

    dataframe["Day_Sin"] = sin_transformer(360).fit_transform(dataframe[["Day"]])
    dataframe["Day_Cos"] = cos_transformer(360).fit_transform(dataframe[["Day"]])

    dataframe["Hour_Sin"] = sin_transformer(24).fit_transform(dataframe[["Hour"]])
    dataframe["Hour_Cos"] = cos_transformer(24).fit_transform(dataframe[["Hour"]])

    dataframe["Min_Sin"] = sin_transformer(60).fit_transform(dataframe[["Minute"]])
    dataframe["Min_Cos"] = cos_transformer(60).fit_transform(dataframe[["Minute"]])

    dataframe["Second_Sin"] = sin_transformer(60).fit_transform(dataframe[["Second"]])
    dataframe["Second_Cos"] = cos_transformer(60).fit_transform(dataframe[["Second"]])

feature_enginerring(df)
def variable_fix(df):
    df = df.drop(0, axis=0)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d/%m/%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df = df.drop(['Date', 'Time'], axis=1)
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())
    return df
df=variable_fix(df)

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

batch_size = 2048
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

        self.fc1 = nn.Linear(input_dim, 22)
        init.xavier_uniform_(self.fc1.weight)
        self.dropout1 = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(22)

        self.fc2 = nn.Linear(22, 20)
        init.xavier_uniform_(self.fc2.weight)
        self.dropout2 = nn.Dropout(0.4)
        self.batch_norm2 = nn.BatchNorm1d(20)

        self.fc3 = nn.Linear(20, 18)
        init.xavier_uniform_(self.fc3.weight)
        self.dropout3 = nn.Dropout(0.4)
        self.batch_norm3 = nn.BatchNorm1d(18)

        self.fc4 = nn.Linear(18, 15)
        init.xavier_uniform_(self.fc4.weight)
        self.dropout4 = nn.Dropout(0.2)
        self.batch_norm4 = nn.BatchNorm1d(15)

        self.fc5 = nn.Linear(15, 12)
        init.xavier_uniform_(self.fc5.weight)
        self.dropout5 = nn.Dropout(0.2)
        self.batch_norm5 = nn.BatchNorm1d(12)

        self.fc6 = nn.Linear(12, 5)
        init.xavier_uniform_(self.fc6.weight)
        self.dropout6 = nn.Dropout(0.2)
        self.batch_norm3 = nn.BatchNorm1d(5)

        self.fc7 = nn.Linear(5, 1)
        init.xavier_uniform_(self.fc4.weight)


    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)

        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)

        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)

        x = torch.relu(self.fc6(x))
        x = self.dropout6(x)

        x = torch.relu(self.fc7(x))

        return x

torch.manual_seed(42)
model = linearRegression(input_dim=24)
model = model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
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
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()

plt.savefig('training_loss_plot.png')

print("Model Summary:")
print(model)

torch.save(model.state_dict(), 'linear_regression_model.pth')
model.load_state_dict(torch.load('linear_regression_model.pth'))
print(model.eval())

model.eval()
predictions = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        y_pred = model(batch_X)
        predictions.append(y_pred.cpu().numpy())


predictions = np.concatenate(predictions)


y_test_numpy = y_test.cpu().numpy()
y_test_sample = y_test_numpy[:1000]
predictions_sample = predictions[:1000]

df_dates = df['Day'][:1000]

def test_vs_prediction(df_dates, y_test_sample, predictions_sample, save_path=None):
    plt.figure(figsize=(10, 5))

    plt.plot(df_dates, y_test_sample, label='Test Verileri', marker='o')
    plt.plot(df_dates, predictions_sample, label='Tahmin Verileri', marker='x')

    plt.title('Test Verileri ve Tahmin Verileri Karşılaştırması')
    plt.xlabel('Tarih')
    plt.ylabel('Değerler')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

test_vs_prediction(df_dates, y_test_sample, predictions_sample, save_path='Six_fully_connected_prediction_vs_Test.png')





