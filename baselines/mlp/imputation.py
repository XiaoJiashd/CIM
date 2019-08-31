from keras.models import Sequential
from keras.layers import Dense
import os
import numpy as np

def train(hiddenDim,trainData,trainLabel,windowSize,batch_Size):
    model.add(Dense(hiddenDim,activation='relu',input_dim=2*windowSize))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer='sgd',
            loss='mse',
            metrics=['mae'])
    model.fit(trainData,trainLabel,epochs=50,batch_size=batch_Size)

proPath = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))
dataPath = os.path.join(proPath, 'data')
trainData=np.loadtxt(os.path.join(dataPath,'trainSet_mlp.csv'))
trainLabel=np.loadtxt(os.path.join(dataPath,'trainLabel_mlp.csv'))
testData=np.loadtxt(os.path.join(dataPath,'testSet_mlp.csv'))
testLabel=np.loadtxt(os.path.join(dataPath,'testLabel_mlp.csv'))
testLabel=testLabel.reshape(testLabel.size,1)

hiddenDimList =[256]
batche_perList=[0.05]
window_size=12
for hiddenDim in hiddenDimList:
    for batche_per in batche_perList:
        batche_Size = int(len(trainData) * batche_per)
        model = Sequential()
        train(hiddenDim,trainData,trainLabel,window_size,batche_Size)
        evluation=model.predict(testData)
        testLabel=testLabel*5
        evluation=evluation*5
        print('RMSE: '+str(np.sqrt(np.mean(np.square((testLabel-evluation))))))
        print('MAE: ' + str(np.mean(np.abs((testLabel - evluation)))))