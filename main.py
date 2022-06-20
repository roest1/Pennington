# pytorch mlp for regression
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from torch.optim import Adam
from torch.nn import ReLU
import numpy as np


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        dataframe = read_csv(path, header=None, usecols=range(7, 52))
        dataframe = dataframe[1:]
        #dataframe.drop(dataframe.tail(116).index, inplace=True)
        dataframe.fillna(0, inplace=True)

        # store the inputs and outputs
        self.X = dataframe.values[:, 1:].astype('float32')
        self.y = dataframe.values[:, 1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition

class MLP(Module):
    # define model elements
    def __init__(self, num_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(num_inputs, 80)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(80, 50)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(50, 1)
        xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        return X


# prepare the dataset
def prepare_data(path):
    # load the dataset
    train = CSVDataset(path)

    # prepare data loaders
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    #test_dataloader = DataLoader(train, batch_size=1024, shuffle=False)
    return train_dataloader  # test_dataloader


# train the model
def train_model(train_dataloader, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    # enumerate epochs
    for epoch in range(20000):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dataloader):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            y_hat = model(inputs)
            # calculate loss
            loss = criterion(y_hat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dataloader, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dataloader):
        # evaluate the model on the test set
        y_hat = model(inputs)
        # retrieve numpy array
        y_hat = y_hat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(y_hat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mean squared error
    # mean_squared_error = mean_squared_error(actuals, predictions)
    # return mean_squared_error
    return mean_squared_error(actuals, predictions)

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    y_hat = model(row)
    # retrieve numpy array
    y_hat = y_hat.detach().numpy()
    return y_hat

# prepare the data - give the path to csv file, Used All DXA sheet from excel file
train_path = "/Users/rileyoest/Pennington/MLP_Regression/Chopped_All_Training_Data.csv"
test_path = "/Users/rileyoest/Pennington/MLP_Regression/Chopped_All_Testing_Data.csv"
train_data = read_csv(train_path, header=None, usecols=range(8, 52))
test_data = read_csv(test_path, header=None, usecols=range(8, 52))
test_data = test_data.fillna(0)

train_dataloader = prepare_data(train_path)
test_dataloader = prepare_data(test_path)
print(len(train_dataloader.dataset), len(test_dataloader.dataset))

# define the network

model = MLP(44)
# train the model
train_model(train_dataloader, model)
# evaluate the model

count = 0
mean = 0
sum_relative_error = 0
sum_squared_error = 0
sum_error = 0
for i in range(1, test_data.shape[0]):
    row = test_data.loc[[i]].astype('float32')
    row = row.values.tolist()
    if row[0][0] != 0:
        y_hat = predict(row, model)
        count += 1
        error = abs(y_hat-row[0][0])
        sum_error += error
        relative_error = error/y_hat
        sum_relative_error += relative_error
        sum_squared_error += error*error
        print('Actual Value: %.3f' % row[0][0], 'Predicted: %.3f' % y_hat,
              'Error: %.3f' % error, 'Relative Error: %.3f' % relative_error)
              
mean_error = sum_error/count
sum_square = 0
for i in range(1, test_data.shape[0]):
    row = test_data.loc[[i]].astype('float32')
    row = row.values.tolist()
    if row[0][0] != 0:
        y_hat = predict(row, model)
        error = abs(y_hat - row[0][0])
        sum_square += (error-mean_error)*(error-mean_error)


print('Mean Error: %.3f' % mean_error)
print('Standard Deviation on Errors: %.3f' % sqrt(sum_square/count))
print('Mean Relative Error: %.3f' % (sum_relative_error/count))
print('Mean Squared Error: %.3f' % (sum_squared_error/count))
print('Root Mean Squared Error: %.3f' % sqrt(sum_squared_error/count))
