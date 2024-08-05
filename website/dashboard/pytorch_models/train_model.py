import numpy as np
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable 
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

def split_train_test(data,training_split = 0.8):
    data = data.copy()
    X = data.drop(columns = ['Close'])
    y = data[['Close']]

    X_lagged = X.shift(1).dropna()
    y = y.iloc[1:]

    training_percentage = int(training_split * len(X_lagged))

    features_used = ','.join(X_lagged.columns)
    num_of_features = len(X_lagged.columns)

    mm = MinMaxScaler()
    ss = StandardScaler()
    X_ss = ss.fit_transform(X_lagged)
    y_mm = mm.fit_transform(y)

    X_train = X_ss[:training_percentage, :]
    X_test = X_ss[training_percentage:, :]

    y_train = y_mm[:training_percentage, :]
    y_test = y_mm[training_percentage:, :] 

    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test)) 


    X_train_tensors = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

    return X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors,mm,features_used,num_of_features

def train(async_data,data,progress_recorder,progress_counter,progress_total,num_epochs):
    torch.manual_seed(1)
    data_stock = data.copy()
    learning_rate = 0.0001 #0.001 lr
    
    training_split = 0.8
    X_train, y_train, X_test, y_test,scaler,features_used,num_of_features = split_train_test(data_stock, training_split = training_split)

    input_size = num_of_features #number of features
    hidden_size = 50 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers

    num_classes = 1 #number of output classes 

    model = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train.shape[1])
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    patience = 10000

    train_loss_array = []
    test_loss_array = []
    learning_rates = []
    models = []
    epochs_with_no_improvements = 0
    best_loss = float('infinity')
    best_model = None

    for epoch in range(num_epochs):
        if async_data.is_aborted():
            break
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train) #forward pass
        
        # obtain the loss function
        loss = criterion(outputs, y_train)
        
        loss.backward() #calculates the loss of the loss function
        
        optimizer.step() #improve from loss, i.e backprop
        train_loss_array.append(loss.item())
        models.append(model)

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            loss_test = criterion(y_test_pred, y_test)
            test_loss_array.append(loss_test.item())

        scheduler.step(loss_test)

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if loss_test < best_loss:
            best_loss = loss_test
            epochs_with_no_improvements = 0
            best_model = model
        else:
            epochs_with_no_improvements += 1

        if epoch % 200 == 0:
            progress_recorder.set_progress(progress_counter, progress_total,description='Training Model ...')
        progress_counter+= 1

        if epochs_with_no_improvements == patience:
            break

        model.train()

    if epochs_with_no_improvements == patience:
        progress_total = progress_counter + 1 + 78
    progress_recorder.set_progress(progress_counter, progress_total,description='Training Model ...')

    best_model.eval()
    with torch.no_grad():
        test_y_pred = best_model(X_test)
        train_x_pred = best_model(X_train)

    true_pred_close_test = scaler.inverse_transform(test_y_pred)
    y_test_unscaled = scaler.inverse_transform(y_test)
    true_pred_close_train = scaler.inverse_transform(train_x_pred)

    print(X_test.shape)
    print(X_test[-1].shape)
    print(X_test[-1])
    RMSE = round(math.sqrt(metrics.mean_squared_error(y_test_unscaled.flatten(), true_pred_close_test.flatten())),5)


    dates = [x.strftime('%d-%m-%Y') for x in data_stock.index]

    training_percentage = int(training_split * len(data_stock))+1
    dates_train_x_axis = dates[1:training_percentage]
    dates_test_x_axis = dates[training_percentage:]

    y_axis_close_test = list(true_pred_close_test.flatten())
    y_axis_close_train = list(true_pred_close_train.flatten())

    
    return dates_train_x_axis, y_axis_close_train, dates_test_x_axis, y_axis_close_test,progress_counter,progress_total,features_used,RMSE