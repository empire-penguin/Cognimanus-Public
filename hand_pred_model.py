import os
import torch
from torch import utils
import numpy as np
from tqdm import tqdm
import import_data
from sklearn.model_selection import train_test_split
import features
from pyfirmata import Arduino, SERVO, util
from time import sleep


#this is the model we are going to use, it's based off of the model in this article:
#https://www.tandfonline.com/doi/full/10.1080/10255842.2020.1861256#F0006
#
#The features that will be inputted into the model will be the features we get from features.py

NUM_EPOCHS = 3000

class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(4+650+650, 50) #5 features: mean of fft, mean, SD, var, and energy
        self.fc2 = torch.nn.Linear(50, 40)
        self.fc3 = torch.nn.Linear(40, 36)
        self.fc4 = torch.nn.Linear(36, 7) #will be 7 outputs for the 7 movements we're classifying

    def forward(self, x):
        #print(x.shape)
        x = torch.sigmoid(self.fc1(x))
        #print(x.shape)
        x = torch.sigmoid(self.fc2(x))
       # print(x.shape)
        x = torch.sigmoid(self.fc3(x))
       # print(x.shape)
        x = torch.sigmoid(self.fc4(x))
        return x

def moveFinger(pin):
    for i in range(0,90):
        board.digital[pin].write(i)
        sleep(0.005)
    for i in range(90,0,-1):
        board.digital[pin].write(i)
        sleep(0.005)
    


def get_data(dataset):

    global X_train
    global X_test
    global y_train
    global y_test
    
    #takes numpy array, splits it into training and testing data and labels,
    # then converts them into tensors
    np_data = features.get_features(dataset)

    X = np_data[:,7:]
    y = np_data[:,0:7]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)
    X_test_1 = torch.from_numpy(X_test)
    X_test_1 = X_test_1.to(torch.float32) 
    #note: article has 70% training data, 15% validation and 15% testing, but not sure how 
    # to work validation data into the model so it's just training and testing for now

model = ANN()
def main(dataset):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    get_data(dataset)

    X_train_torch = torch.from_numpy(X_train)
    X_train_torch = X_train_torch.to(torch.float32) 
    #for fft and energy, they'll be arrays so find a way to convert each array value to float32 
    y_train_torch = torch.from_numpy(y_train)
    y_train_torch = y_train_torch.type(torch.float32)

    #print(model.fc1.weight.shape, model.fc2.weight.shape, model.fc3.weight.shape, model.fc4.weight.shape)
    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        # Forward pass

        y_pred = model(X_train_torch)
        

        # Compute and print loss
        loss = criterion(y_pred, y_train_torch)
        print(f'Epoch {epoch + 1}/3000 | Loss: {loss.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    X_test_torch = torch.from_numpy(X_test) 
    X_test_torch = X_test_torch.to(torch.float32)
    y_test_torch = torch.from_numpy(y_test)
    y_test_torch = y_test_torch.to(torch.float32)
    y_pred = model(X_test_torch)
    
    #print(y_pred)

    # Print the model's predictions
    print('After training:')
    for i in range(X_test_torch.shape[0]):
        
        print(f'Input: {X_test_torch[i].numpy()} | Ground Truth: {y_test_torch[i].numpy()} | Prediction: {y_pred[i].detach().numpy()}')

    #put in way to save model
    torch.save(model.state_dict(), 'model.pt')
    
Fin_Model = ANN()
Fin_Model.load_state_dict(torch.load('model.pt'))
model.eval()


def run(dataset):
    #movement will be a 1D array of data from import data 
    #
    y_pred = Fin_Model(dataset)
    classifications = []

    for i in y_pred:
        
        m = torch.nn.Softmax(dim=0)
        i_m = m(i.float())
        value_of_largest = torch.max(i_m)
        onehot_output = []
        for x in i_m:
            if x == value_of_largest:
                onehot_output.append(1)
            else:
                onehot_output.append(0)
        #onehot_output = [x for x in xx_m if x == value_of_largest]
        #print(value_of_largest, i_m, onehot_output)
        onehot_output = np.asarray(onehot_output)
        classifications.append(np.argmax(onehot_output))

        #print(np.argmax(onehot_output))
    #return [3,4,1,2,3,5]
    return classifications
   

if __name__ == '__main__':
    #main(import_data.all_data)
    get_data(import_data.all_data)
    state_list = run(X_test)
    for state in state_list:

   
        # from Variables import l,m,r

        # def findValue(left_bit, middle_bit, right_bit):
        #     value= left_bit*(2**2) + middle_bit*(2**1) + right_bit*(2**0)
        #     return value
        #state = run()
        port = '/dev/serial/by-id/usb-Arduino_Srl_Arduino_Uno_75437303730351818022-if00'
        board = Arduino(port)

        # servo0 = 9 #Thumb
        servo1 = 5 #Index 
        servo2 = 6 #Middle
        servo3 = 10 #Ring
        servo4 = 9 #Pinkie
        # servo5 = 7 #Wrist


        #board.digital[servo0].mode = SERVO
        board.digital[servo1].mode = SERVO
        board.digital[servo2].mode = SERVO
        board.digital[servo3].mode = SERVO
        board.digital[servo4].mode = SERVO
        # board.digital[servo5].mode = SERVO


        print(f'servo state: {state}')
        #moveFinger(servo1)
        #moveFinger(servo2)
        #moveFinger(servo3)
        #moveFinger(servo4)

        moveFinger(vars()['servo'+str(state)])
