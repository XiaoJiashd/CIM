# _*_ coding:utf-8 _*_
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense

daywin_size=5
window_size=12

trainSample_number=967209 #t=5 /0.4/0.6/0.8
testSample_number=241802  #t=5
# trainSample_number=476805 #t=10
# testSample_number=119201 #t=10
# trainSample_number=331315 #t=15
# testSample_number=82828 #t=15

# trainSample_number=386884 #0.4
# trainSample_number=580326 #0.6
# trainSample_number=773768 #0.8

proPath = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))
dataPath = os.path.join(proPath, 'data')
trainData_temp=np.loadtxt(os.path.join(dataPath,'trainSet_cnn.csv')).reshape((trainSample_number,daywin_size,window_size*2))
trainLabel=np.loadtxt(os.path.join(dataPath,'trainLabel_cnn.csv'))
testData_temp=np.loadtxt(os.path.join(dataPath,'testSet_cnn.csv')).reshape((testSample_number,daywin_size,window_size*2))
testLabel=np.loadtxt(os.path.join(dataPath,'testLabel_cnn.csv'))
testLabel=testLabel.reshape(testLabel.size,1)

trainData=np.zeros((trainSample_number,daywin_size,window_size*2,1))
testData=np.zeros((testSample_number,daywin_size,window_size*2,1))
for i in range(trainData_temp.shape[0]):
    temp=trainData_temp[i]
    temp=temp.reshape(daywin_size,window_size*2,1)
    trainData[i]=temp
for i in range(testData_temp.shape[0]):
    temp=testData_temp[i]
    temp=temp.reshape(daywin_size,window_size*2,1)
    testData[i]=temp
filterList=[(2,2)]
dim=16
batche_perList=[0.02]
for filter in filterList:
    for batche_per in batche_perList:
        batche_Size = int(len(trainData) * batche_per)
        model = Sequential()
        model.add(Conv2D(dim,filter,activation='relu',input_shape=(daywin_size,window_size*2,1)))
        model.add(Flatten())
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='sgd',
                      loss='mse',
                      metrics=['mae'])
        model.fit(trainData, trainLabel, epochs=50, batch_size=batche_Size)
        evluation = model.predict(testData)
        testLabel = testLabel * 5
        evluation = evluation * 5
        tL_temp = testLabel[testLabel > 0.5]
        ev_temp = evluation[testLabel > 0.5]
        print('RMSE: ' + str(np.sqrt(np.mean(np.square((testLabel - evluation))))))
        print('MAE: ' + str(np.mean(np.abs((testLabel - evluation)))))
        print('MRE: ' + str(np.mean(np.abs(tL_temp - ev_temp) / tL_temp)))