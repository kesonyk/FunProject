from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.utils.visualize_util import plot

from keras.models import Model
import keras
from keras.datasets.data_utils import get_file
import keras
from keras.layers import Input,Convolution2D,Activation,Flatten,MaxPooling2D,Dropout,Dense

import numpy as np
import sys
import os
import cPickle
import time

batch_size = 1024
nb_fine_classes = 100
nb_coarse_classes=20
nb_epoch=100
nb_each_epoch = 1
data_augmentation = False

cwd=os.getcwd()
savePath=cwd+"/result/"
if not os.path.exists(savePath):
    os.mkdir(savePath)

def load_batch_2label(fpath):
    f=open(fpath,"rb")
    if sys.version_info<(3,):
        d=cPickle.load(f)
    else:
        d=cPickle.load(f,encoding="bytes")
        # decode utf8
        for k,v in d.items():
            del(d[k])
            d[k.encode("utf8")]=v
    f.close()
    data=d["data"]
    coarse_labels=d["coarse_labels"]
    fine_labels=d["fine_labels"]

    data=data.reshape(data.shape[0],3,32,32)
    return data,coarse_labels,fine_labels

def load_data_2label():
    dirname="cifar-100-python"
    origin="http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    path=get_file(dirname,origin=origin,untar=True)

    nb_test_samples=10000
    nb_train_samples=50000

    fpath=os.path.join(path,"train")
    X_train,coarseLabel_train,fineLabel_train=load_batch_2label(fpath)

    fpath=os.path.join(path,"test")
    X_test,coarseLabel_test,fineLabel_test=load_batch_2label(fpath)
    
    coarseLabel_train=np.reshape(coarseLabel_train,(len(coarseLabel_train),1))
    fineLabel_train=np.reshape(fineLabel_train,(len(fineLabel_train),1))

    coarseLabel_test=np.reshape(coarseLabel_test,(len(coarseLabel_test),1))
    fineLabel_test=np.reshape(fineLabel_test,(len(fineLabel_test),1))

    return (X_train,coarseLabel_train,fineLabel_train),(X_test,coarseLabel_test,fineLabel_test)
    






    
# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train,coarseLabel_train,fineLabel_train),(X_test,coarseLabel_test,fineLabel_test)=load_data_2label()

# convert class vectors to binary class matrices
coarseLabel_train = np_utils.to_categorical(coarseLabel_train, nb_coarse_classes)
coarseLabel_test = np_utils.to_categorical(coarseLabel_test, nb_coarse_classes)
fineLabel_train = np_utils.to_categorical(fineLabel_train, nb_fine_classes)
fineLabel_test = np_utils.to_categorical(fineLabel_test, nb_fine_classes)


#create the mode
def create_model():
    img_input=Input(shape=(3,32,32),name="img_input")
    
    conv1_1=Convolution2D(32,3,3,activation="relu",border_mode="same")(img_input)
    conv1_2=Convolution2D(32,3,3,activation="relu",border_mode="same")(conv1_1)
    pool1_1=MaxPooling2D((2,2),border_mode="same")(conv1_2)
    drop1_1=Dropout(0.5)(pool1_1)

    conv2_1=Convolution2D(64,3,3,activation="relu",border_mode="same")(drop1_1)
    conv2_2=Convolution2D(64,3,3,activation="relu",border_mode="same")(conv2_1)
    pool2_1=MaxPooling2D((2,2),border_mode="same")(conv2_2)
    drop2_1=Dropout(0.5)(pool2_1)
    
    #out1 for coarse label
    flatten3_1=Flatten()(drop1_1)
    dense3_1=Dense(512,activation="relu")(flatten3_1)
    drop3_1=Dropout(0.5)(dense3_1)
    dense3_2=Dense(256,activation="relu")(drop3_1)
    drop3_2=Dropout(0.5)(dense3_2)
    dense3_3=Dense(128,activation="relu")(drop3_2)
    drop3_3=Dropout(0.5)(dense3_3)

    coarse_out=Dense(nb_coarse_classes,activation="softmax",name="coarse_out")(drop3_3)

    #out for fine label
    flatten3_2=Flatten()(drop2_1)
    dense3_2=Dense(512,activation="relu")(flatten3_2)
    drop3_2=Dropout(0.5)(dense3_2)
    fine_out=Dense(nb_fine_classes,activation="softmax",name="fine_out")(drop3_2)

    model=Model(input=img_input,output=[coarse_out,fine_out])
    return model


model=create_model()
model.compile(optimizer='rmsprop',loss="categorical_crossentropy",loss_weights=[1,1],metrics=["accuracy"])
plot(model,to_file="model.png")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

fine_test_acc=np.zeros((nb_epoch,1))
fine_test_loss=np.zeros((nb_epoch,1))

fine_train_acc=np.zeros((nb_epoch,1))


class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.train_acces=[]
    def on_batch_end(self,batch,logs={}):
        self.train_acces.append(logs.get("fine_out_acc"))

history=AccHistory()

if not data_augmentation:
    print("Not using data augmentation or normalization")
    for i in xrange(nb_epoch):
        model.fit(X_train,[coarseLabel_train,fineLabel_train], batch_size=batch_size, nb_epoch=nb_each_epoch,verbose=1,callbacks=[history])
        score = model.evaluate(X_test,[coarseLabel_test,fineLabel_test], batch_size=batch_size)
        print("iter:",i)
        print("score:",score)
        
        fine_train_acc[i,]=np.mean(history.train_acces)
        
        fine_test_acc[i,]=score[4]
        
        fine_test_loss[i,]=score[2]
        
        print("Train acc:",fine_train_acc[i,])
        print('Test acc:',fine_test_acc[i,])
        

        np.save(savePath+"multiTask_fine_train_acc",fine_train_acc)
        np.save(savePath+"multiTask_fine_test_loss",fine_test_loss)
        np.save(savePath+"multiTask_fine_test_acc",fine_test_acc)

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
        fill_mode="nearest")

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for i in xrange(nb_epoch):
        print("Epoch",i)
        batches=0
        fine_train_acc_epoch=[]
        t0=time.time()
        for X_batch,coarseLabel_batch in datagen.flow(X_train,coarseLabel_train,batch_size=batch_size):
            fineLabel_batch=fineLabel_train[batches*batch_size:(batches+1)*batch_size]
            model.fit(X_batch,[coarseLabel_batch,fineLabel_batch],batch_size=batch_size,nb_epoch=1,verbose=0,callbacks=[history])
            fine_train_acc_epoch.append(np.mean(history.train_acces))
            batches+=1
            if batches>=len(X_train)/batch_size:
                break
        
        t1=time.time()
        print("training 1 epoch time:",(t1-t0))
        score=model.evaluate(X_test,[coarseLabel_test,fineLabel_test],batch_size=batch_size)
        
        fine_train_acc[i,]=np.mean(fine_train_acc_epoch)
        fine_test_acc[i,]=score[4]
        fine_test_loss[i,]=score[2]
        
        print("train acc:%f test_acc:%f test_loss:%f"%(fine_train_acc[i,],fine_test_acc[i,],fine_test_loss[i,]))
        np.save(savePath+"multiTask_fine_train_acc_augment",fine_train_acc)
        np.save(savePath+"multiTask_fine_test_loss_augment",fine_test_loss)
        np.save(savePath+"multiTask_fine_test_acc_augment",fine_test_acc)






