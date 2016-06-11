from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,UpSampling2D,Dropout
from keras.models import Model
from keras.datasets import cifar10
import numpy as np

(X_train,_),(X_test,_)=cifar10.load_data()
np.save("X",X_train[0])

input_img=Input(shape=(3,32,32))

conv1=Convolution2D(32,3,3,activation="relu",border_mode="same")(input_img)
conv2=Convolution2D(32,3,3,activation="relu",border_mode="same")(conv1)
pool1=MaxPooling2D((2,2),border_mode="same")(conv2)

conv3=Convolution2D(64,3,3,activation="relu",border_mode="same")(pool1)
conv4=Convolution2D(64,3,3,activation="relu",border_mode="same")(conv3)
pool2=MaxPooling2D((2,2),border_mode="same")(conv2)

encoder=Model(input=input_img,output=pool2)

conv5=Convolution2D(64,3,3,activation="relu",border_mode="same")(pool2)
conv6=Convolution2D(64,3,3,activation="relu",border_mode="same")(conv5)
upsamp1=UpSampling2D((2,2))(conv6)

conv7=Convolution2D(32,3,3,activation="relu",border_mode="same")(upsamp1)
conv8=Convolution2D(32,3,3,activation="relu",border_mode="same")(conv7)
upsamp2=UpSampling2D((2,2))(conv7)

decoded=Convolution2D(3,3,3,activation="sigmoid",border_mode="same")(upsamp2)


print"compile"
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(X_train,_),(X_test,_)=cifar10.load_data()

X_train=X_train.astype("float32")/255
print("X_train shape:",X_train.shape)
X_test=X_test.astype("float32")/255

autoencoder.fit(X_train,X_train,nb_epoch=50,batch_size=256,shuffle=True,validation_data=(X_test,X_test),verbose=1)


decoded_imgs=antoencoder.predict(X_test)
np.save("decoded_img",decoded_imgs)





