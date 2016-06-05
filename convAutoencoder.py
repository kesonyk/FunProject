from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,UpSampling2D,Dropout
from keras.models import Model
from keras.datasets import mnist
import numpy as np


input_img=Input(shape=(1,28,28))

conv1=Convolution2D(16,3,3,activation="relu",border_mode="same")(input_img)
pool1=MaxPooling2D((2,2),border_mode="same")(conv1)
conv2=Convolution2D(8,3,3,activation="relu",border_mode="same")(pool1)
pool2=MaxPooling2D((2,2),border_mode="same")(conv2)
conv3=Convolution2D(8,3,3,activation="relu",border_mode="same")(pool2)
encoded=MaxPooling2D((2,2),border_mode="same")(conv3)

#at this point the representation is (8,4,4)
conv4=Convolution2D(8,3,3,activation="relu",border_mode="same")(encoded)
upsamp1=UpSampling2D((2,2))(conv4)

conv5=Convolution2D(8,3,3,activation="relu",border_mode="same")(upsamp1)
upsamp2=UpSampling2D((2,2))(conv5)

conv6=Convolution2D(16,3,3,activation="relu")(upsamp2)
upsamp3=UpSampling2D((2,2))(conv6)

decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(upsamp3)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

autoencoder.fit(x_train, x_train,nb_epoch=50,batch_size=128,shuffle=True,
				validation_data=(x_test, x_test))
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



















