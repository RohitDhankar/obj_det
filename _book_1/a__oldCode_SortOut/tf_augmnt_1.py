

import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile
import cv2
import os 
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#FOOBAR_Dhankar? is this the same as above ??
#Import ImageDataGenerator for Loading Images
#TODO - from keras.preprocessing.image import ImageDataGenerator
## SOURCE >> https://keras.io/api/preprocessing/image/


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy') > .98) & (logs.get('val_accuracy') > .9):
            print("Reached 98% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

train_datagen = ImageDataGenerator(rescale=1/255)
print("---type(train_datagen)------",type(train_datagen))
validation_datagen = ImageDataGenerator(rescale=1/255)

input_dir = '../input_dir/'#airplanes/'
train_generator = train_datagen.flow_from_directory(
        input_dir,  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 150x150
        batch_size=128, # number of instances of DATA - that are evaluated before a weight update in the network - this is batch_size .
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
        input_dir,  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

a = train_generator.class_indices
print("train_dataset.class_indices---",a)

from tensorflow.keras.optimizers import RMSprop

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=1e-3), #learning_rate=1e-3 ## Earlier >> lr=0.001
              metrics=['accuracy'])

              #SOURCE -- >> https://www.tensorflow.org/guide/keras/train_and_evaluate
              # The metrics argument should be a list -- your model can have any number of metrics.

#

history = model.fit(
      train_generator,
      steps_per_epoch=1,  
      epochs=20,
      validation_data = validation_generator,callbacks=[callbacks])

#
#evaluation_of_model = model.evaluate(train_generator,validation_data = validation_generator,callbacks=[callbacks])
evaluation_of_model = model.evaluate(train_generator,validation_generator)
print("-----evaluation_of_model----",type(evaluation_of_model))

# Traceback (most recent call last):
#   File "tf_augmnt_1.py", line 92, in <module>
#     evaluation_of_model = model.evaluate(train_generator,validation_generator)
#   File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/keras/utils/traceback_utils.py", line 67, in error_handler
#     raise e.with_traceback(filtered_tb) from None
#   File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/keras/engine/data_adapter.py", line 915, in __init__
#     raise ValueError("`y` argument is not supported when using "
# ValueError: `y` argument is not supported when using `keras.utils.Sequence` as input.









model.save("tf_model_augment_5")
img_md = tf.keras.preprocessing.image
model = tf.keras.models.load_model('tf_model_augment_5')
print("----------type(model--------",type(model))
# # test_img = img_md.load_img("../test_imgs/image_0020.jpg", target_size=(224, 224))
# # test_img_array = img_md.img_to_array(test_img)
# # test_img_array = test_img_array.reshape(1, 224, 224, 3)
# # pred_val = model.predict(test_img_array)
# # print("------pred_val----",pred_val)
# # print("------pred_val.argmax()-----",pred_val.argmax())



### OK HOLD TESTING --- evaluation_of_model
# dir_path = "../test_imgs/test"
# for i in os.listdir(dir_path):
#     img = image.load_img(dir_path+ '//' + i , target_size=(200, 200))
#     #plt.imshow(img)
#     print("--Test image_name ==",i)
#     #plt.show()
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])

#     classes =model.predict(images)
#     if classes[0]==0:
#         print("--got image Class__airplanes_0___")
#     else:
#         print("--got image Class__not_airplanes_1___")





# # img_md = tf.keras.preprocessing.image
# # img = img_md.load_img("images_products/img_1.png")
# # img = img_md.img_to_array(img)

# # ImageDataGen = tf.keras.preprocessing.image.ImageDataGenerator
# # print("---type(ImageDataGen)------",type(ImageDataGen))

# # train = ImageDataGen(rescale=1. / 255)
# # print("---type(train)------",type(train))
# # validation = ImageDataGen(rescale=1. / 255)
# # print("---type(validation)------",type(validation))
# # ---type(ImageDataGen)------ <class 'type'>
# # ---type(train)------ <class 'keras.preprocessing.image.ImageDataGenerator'>
# # ---type(validation)------ <class 'keras.preprocessing.image.ImageDataGenerator'>

# input_dir = '../input_dir/'#airplanes/'

# train_dataset = train.flow_from_directory(input_dir, target_size=(224, 224),batch_size=100,class_mode = 'binary')#shuffle=True)
# print("----train_dataset----",train_dataset) # Found 875 images belonging to 2 classes.
# validation_dataset = train.flow_from_directory(input_dir, target_size=(224, 224),batch_size=100,class_mode = 'binary')#shuffle=True)
# print("----validation_dataset----",validation_dataset) # Found 875 images belonging to 2 classes.

# a = train_dataset.class_indices
# print("train_dataset.class_indices---",a)
# b = validation_dataset.class_indices
# print("validation_dataset.class_indices---",b)
# train_cls = train_dataset.classes
# #print("train_cls.classes---",train_cls)
# vald_cls = validation_dataset.classes
# #print("vald_cls.classes---",vald_cls)
# #

# from tensorflow.keras.optimizers import RMSprop

# model = tf.keras.models.Sequential([
#     # Note the input shape is the desired size of the image 150x150 with 3 bytes color
#     # This is the first convolution
    
#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),## FOOBAR# input_shape=(200, 200, 3)),
    
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The third convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The fourth convolution
#     #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     #tf.keras.layers.MaxPooling2D(2,2),
#     # The fifth convolution
#     #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     #tf.keras.layers.MaxPooling2D(2,2),
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])


# ### FOOBAR
# # model.compile(loss='binary_crossentropy',
# #               optimizer=RMSprop(lr=0.001),
# #               metrics=['accuracy'])


# ### .get_shape().as_list()[0]


# # model = tf.keras.Sequential([
# #     tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',input_shape=(224, 224, 3)),
# #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
# #     tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
# #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(64, activation='relu'),
# #     tf.keras.layers.Dense(2, activation='softmax')
# # ])
# # model.compile(
# #     optimizer=tf.keras.optimizers.Adam(lr=0.0001),
# #     loss="categorical_crossentropy",# binary_crossentropy
# #     metrics=['accuracy']
# # )


# print(model.summary())

# print("------train_dataset.samples-------",train_dataset.samples) # 875 
# # TRAIN DATA Count == 875

# #
# history = model.fit(train_dataset,
#                     steps_per_epoch=1, #train_dataset.samples
#                     epochs=20,
#                     validation_data=validation_dataset,
#                     #validation_steps=validation_dataset.samples,
#                     verbose=1
#                     )


# # history = model.fit(
# #       train_generator,
# #       steps_per_epoch=1,  
# #       epochs=20,
# #       validation_data = validation_generator,callbacks=[callbacks])


# print("--------history-------",history) #https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History
# print("--------history.params-------",history.params)
# print("--------history.history.keys()------",history.history.keys())
# #
# import os 
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import zipfile
# import cv2
# import os 
# import numpy as np
# import matplotlib.image as mpimg
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image

# dir_path = "../test_imgs/test"
# for i in os.listdir(dir_path):
#     img = image.load_img(dir_path+ '//' + i , target_size=(200, 200))
#     plt.imshow(img)
#     plt.show()
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])

#     classes =model.predict(images)
#     if classes[0]==0:
#         print("----GOT__0")
#     else:
#         print("----GOT__1")


# # model.save("tf_model_augment_1")
# # img_md = tf.keras.preprocessing.image
# # model = tf.keras.models.load_model('tf_model_augment_1')
# # test_img = img_md.load_img("../test_imgs/image_0020.jpg", target_size=(224, 224))
# # test_img_array = img_md.img_to_array(test_img)
# # test_img_array = test_img_array.reshape(1, 224, 224, 3)
# # pred_val = model.predict(test_img_array)
# # print("------pred_val----",pred_val)
# # print("------pred_val.argmax()-----",pred_val.argmax())



# # if __name__ == "__main__":
# #     import tensorflow
# #     import time
# #     import tensorflow as tf
# #     import matplotlib.pyplot as plt
