import tensorflow as tf
import  numpy as np
(train_images,Y_train),(test_images,Y_test) = tf.keras.datasets.mnist.load_data()

train_images= train_images.reshape(60000,28,28,1)
train_images = train_images/255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer = 'adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')
model.fit(train_images, Y_train, epochs=10,batch_size=128,verbose=1,validation_data=(test_images,Y_test))
print("The model has successfully trained")
model.save('C:/Users/Yoga/Desktop/PY/Models/mnist.h5')
print("Saving the model as mnist.h5")