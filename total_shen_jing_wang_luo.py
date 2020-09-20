from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD, Adam, RMSprop

from keras.utils import np_utils

from sklearn import preprocessing

#第一步：调用pandas包
import pandas as pd
#第二步：读取数据
io=r'C:\Users\Administrator\Desktop\data\total_train.xlsx'
##表格1
data = pd.read_excel(io,header=0)#读入数据文件关键中的关键：header=None

df=pd.DataFrame(data)



shuffle_data=df.sample(frac=1)#行乱序





x_train=shuffle_data.iloc[:,1:21]#训练多维特征
y_train=shuffle_data.loc[:,'label']#训练标签

x_train_0=preprocessing.minmax_scale(x_train,feature_range=(-1,1))#归一化

x_train_1,x_test_1,y_train_1,y_test_1=train_test_split(x_train_0,y_train,test_size=0.30,random_state=30)
#
y_train_1=np_utils.to_categorical(y_train_1,12)#矢量编码
y_test_1=np_utils.to_categorical(y_test_1,12)


model=Sequential()
model.add(Dense(512,input_shape=(20,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(12))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history=model.fit(x_train_1,y_train_1,
                  batch_size=128,epochs=5)