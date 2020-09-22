import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy 
from keras.optimizers import Adam, Nadam
from keras.models import load_model
from tensorflow.keras import optimizers
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
K.common.set_image_dim_ordering('th')

def cnn_model():
    num_labels = 3
    def swish_activation(x):
        return (K.sigmoid(x) * x)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(48,48,1)))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation=swish_activation))
    model.add(Dropout(0.4))
    model.add(Dense(num_labels, activation='softmax'))
    return model

def train_a_model(trainfile):
    df=pd.read_csv(trainfile)
    df['emotion']=df['emotion'].astype('category')
    df['emotion'] = df['emotion'].cat.codes
    
    x_train=df.iloc[:,1:]
    y_train=df.iloc[:,:1]
    num_labels = 3
    x_train = np.array(x_train,'float32')
    y_train=np_utils.to_categorical(y_train, num_classes=num_labels)

    x_train -= np.mean(x_train, axis=0)
    x_train /= np.std(x_train, axis=0)
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)

    datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=10,  
            zoom_range = 0.0,  
            width_shift_range=0.1,  
            height_shift_range=0.1,  
            horizontal_flip=False, 
            vertical_flip=False)  

    datagen.fit(x_train)

    num_features = 64
    num_labels = 3
    batch_size = 32
    epochs =20
    width, height = 48, 48
    
    model=cnn_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7) , metrics=['accuracy'])
    steps_per_epoch = len(x_train) // batch_size
    
    lr_reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.1, min_delta=0.001, patience=1, verbose=1)
    
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[lr_reduce],
                    epochs = epochs, verbose = 0)
    return model
    pass

def test_the_model(testfile, model):
    
    df=df=pd.read_csv(testfile)

    df['emotion']=df['emotion'].astype('category')
    df['emotion'] = df['emotion'].cat.codes
    p_check=list(df['emotion'])
    x_test=df.iloc[:,1:]
    y_test=df.iloc[:,:1]
    num_labels = 3
    x_test = np.array(x_test,'float32')

    y_test=np_utils.to_categorical(y_test,    num_classes=num_labels)

    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)

    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

    test_results = model.evaluate(x_test,y_test, verbose=0)
    
    y_pred = model.predict(x_test)
    a=np.argmax(y_pred, axis=1)
    dic = {0:'fear', 1:'happy', 2:'sad'}
    result=([dic.get(n, n) for n in a])
    return result