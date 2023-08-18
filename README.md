# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![deep](https://github.com/KeerthikaNagarajan/basic-nn-model/assets/93427089/f3e11dc0-b48d-4a90-80e9-27564bb3af66)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

**Developed By:** Keerthika N

**Register Number:** 212221230049

```python
### Importing Modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

### Authenticate & Create Dataframe using Data in Sheets

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MyMLData').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()

### Assign X and Y values

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X
y

### Normalize the values & Split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

### Create a Neural Network & Train it

#Create the model
ai=Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])

#Compile the model
ai.compile(optimizer='rmsprop',loss='mse')

#fit the model
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)

### Plot the Loss

loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

### Evaluate the model

X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

### Predict for some value

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)

```
## Dataset Information

<img width="170" alt="data" src="https://github.com/KeerthikaNagarajan/basic-nn-model/assets/93427089/522827b7-92ad-4a4f-96d1-99e13d01175f">

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="421" alt="1" src="https://github.com/KeerthikaNagarajan/basic-nn-model/assets/93427089/612168a4-56ed-4de0-9043-e0ca4472ba96">

### Test Data Root Mean Squared Error

<img width="408" alt="2" src="https://github.com/KeerthikaNagarajan/basic-nn-model/assets/93427089/1c39dd91-88dd-4c33-993d-be7dd5ac7106">

### New Sample Data Prediction

<img width="315" alt="3" src="https://github.com/KeerthikaNagarajan/basic-nn-model/assets/93427089/d46acd2e-0218-4b9f-91ba-01b7e5fb3c48">

## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
