"""
This is an example written by the author of keras on his blog
blog.keras.io
implement a simplest possible autoencoder with a single
fully-connected layer as encoder and as decoder based on mnist
"""

from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#this is the size of our encodded representations
encoding_dim=32

#this is our input placeholder
input_img=Input(shape=(784,))
#"encoded" is the encodeed representation of the input
encoded=Dense(encoding_dim,activation="relu")(input_img)
#"decoded" is the lossy reconstruction of the input
decoded=Dense(784,activation="sigmoid")(encoded)

#this model maps an input to its reconstruction
autoencoder=Model(input=input_img,output=decoded)

#this model maps an input to its encoded representation
encoder=Model(input=input_img,output=encoded)

#crate the decoder model
encoded_input=Input(shape=(encoding_dim,))
decoder_layer=autoencoder.layers[-1]
decoder=Model(input=encoded_input,output=decoder_layer(encoded_input))


#use a per-pixel binary crossentropy loss
autoencoder.compile(optimizer="adadelta",loss="binary_crossentropy")

(x_train,_),(x_test,_)=mnist.load_data()
x_train=x_train.astype("float32")/255.
x_test=x_test.astype("float32")/255.
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=256,shuffle=True,validation_data=(x_test,x_test))

encoded_imgs=encoder.predict(x_test)
decoded_imgs=decoder.predict(encoded_imgs)

n=10    #how many digits we will display
plt.figure(figsize=(20,4))
for i in xrange(n):
    #display original
    ax=plt.subplot(2,n,i)
    plt.imshow(x_test[i].reshape(28,28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax=plt.subplot(2,n,i+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()















