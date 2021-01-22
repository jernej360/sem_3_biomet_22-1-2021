import glob
import os
import json
import shutil


def copyTo(destination,files):
    for file in files:
        print("copying from: "+file+"    to:"+destination+os.path.basename(file))
        os.makedirs(os.path.dirname(destination+os.path.basename(file)), exist_ok=True)
        os.system('cp '+file + ' ' + destination+os.path.basename(file))

def main():
    files = glob.glob('awe/train/*/*.json', recursive=True)
    i=0
    for jsonFile in files:
        with open(jsonFile) as f:
            paresedTmp = json.load(f)
            print(jsonFile+"   gender: "+paresedTmp["gender"])
            imagefiles = glob.glob(os.path.dirname(jsonFile)+'/*.png', recursive=True)
            copyTo("datasets/"+paresedTmp["gender"]+"/"+str(i)+"/",imagefiles)
            i=i+1


    print("finish")


if __name__ == "__main__":
    main()


# def readFolders(path,label):
#     training_data = []
#     files = glob.glob(path+'/*.png', recursive=True)
#     for imagePath in tqdm(files):
#         img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
#         training_data.append([np.array(img),np.array(label)])
#     return training_data
#
# def create_train_data(retrain=False):
#     if retrain:
#         training_data = []
#         training_data.append(readFolders(TRAIN_DIR + "/m/", [1, 0]))
#         training_data.append(readFolders(TRAIN_DIR + "/f/", [0, 1]))
#         shuffle(training_data)
#         np.save('train_data.npy', training_data)
#     else:
#         training_data = np.load('train_data.npy',allow_pickle=True)
#     print("finished getting data")
#     return training_data



#/////////////////////////////////////////////////////////////////////////////////////////////
# import cv2
# import numpy as np
# from random import shuffle
# from tqdm import tqdm
# import glob
# import tensorflow as tf
# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
# import os
#
# TRAIN_DIR = 'datasets/train'
# TEST_DIR = 'datasets/test'
# IMG_SIZE = 50
# LR = 1e-3
#
# MODEL_NAME = 'maleVsFemale-{}-{}.model'.format(LR, '2conv-basic')
#
# #  males  ===> [1,0]
# #  female ===> [0,1]
# CATEGORIES = ["m", "f"]
#
#
# def read_data():
#     training_data = []
#     for category in CATEGORIES:
#         path = os.path.join(TRAIN_DIR, category)
#         class_num = CATEGORIES.index(category)
#         print("\n"+str(class_num)+". reading "+path)
#         for img in tqdm(os.listdir(path)):
#             img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#             new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#             training_data.append([new_array, class_num])
#
#     print("finished reading images")
#     return training_data
#
#
# def preproces(resample=False):
#     if resample:
#         training_data = read_data()
#         shuffle(training_data)
#         np.save('train_data.npy', training_data)
#     else:
#         training_data = np.load('train_data.npy',allow_pickle=True)
#     print("finished gathering data")
#     return training_data
#
# def process(train_data):
#     convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
#
#     convnet = conv_2d(convnet, 32, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)
#
#     convnet = conv_2d(convnet, 64, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)
#
#     convnet = fully_connected(convnet, 1024, activation='relu')
#     convnet = dropout(convnet, 0.8)
#
#     convnet = fully_connected(convnet, 2, activation='softmax')
#     convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
#
#     model = tflearn.DNN(convnet, tensorboard_dir='log')
#     if os.path.exists('{}.meta'.format(MODEL_NAME)):
#         model.load(MODEL_NAME)
#         print('model loaded!')
#
#     train = train_data[:-500]
#     test = train_data[-500:]
#
#     X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#     Y = [i[1] for i in train]
#
#     test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#     test_y = [i[1] for i in test]
#
#     model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), run_id=MODEL_NAME)
#
#     model.save(MODEL_NAME)
#     print("finished processing")
#
#
# def main():
#     data=preproces()
#     process(data)
#     print("finish")
#
# if __name__ == '__main__':
#     main()



#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# import numpy as np
# import pandas as pd
# from keras.preprocessing.image import ImageDataGenerator,load_img
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import random
# import tensorflow as tf
# import os
# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPooling2D,\
#      Dropout,Flatten,Dense,Activation,\
#      BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#
#
# Image_Width=50
# Image_Height=50
# Image_Size=(Image_Width,Image_Height)
# Image_Channels=3
#
# def learn(retrain=False):
#     print("running")
#     filenames = os.listdir("./dataset_all")
#
#     categories = []
#     for f_name in filenames:
#         category = f_name.split('.')[0]
#         if category == 'm':
#             categories.append(1)
#         else:
#             categories.append(0)
#
#     df = pd.DataFrame({
#         'filename': filenames,
#         'category': categories
#     })
#
#     model = Sequential()
#
#     model.add(Conv2D(32,(3,3),activation='relu',input_shape=(Image_Width,Image_Height,Image_Channels)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64,(3,3),activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(128,(3,3),activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(512,activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(2,activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy',
#       optimizer='rmsprop',metrics=['accuracy'])
#
#     model.summary()
#
#     earlystop = EarlyStopping(patience=10)
#     learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
#     callbacks = [earlystop, learning_rate_reduction]
#
#     df["category"] = df["category"].replace({0: 'f', 1: 'm'})
#     train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
#
#     train_df = train_df.reset_index(drop=True)
#     validate_df = validate_df.reset_index(drop=True)
#
#     total_train = train_df.shape[0]
#     total_validate = validate_df.shape[0]
#     batch_size = 15
#
#     train_datagen = ImageDataGenerator(rotation_range=15,
#                                        rescale=1. / 255,
#                                        shear_range=0.1,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                        width_shift_range=0.1,
#                                        height_shift_range=0.1
#                                        )
#
#     train_generator = train_datagen.flow_from_dataframe(train_df,
#                                                         "./dataset_all/", x_col='filename', y_col='category',
#                                                         target_size=Image_Size,
#                                                         class_mode='categorical',
#                                                         batch_size=batch_size)
#
#     validation_datagen = ImageDataGenerator(rescale=1. / 255)
#     validation_generator = validation_datagen.flow_from_dataframe(
#         validate_df,
#         "./dataset_all/",
#         x_col='filename',
#         y_col='category',
#         target_size=Image_Size,
#         class_mode='categorical',
#         batch_size=batch_size
#     )
#
#     test_datagen = ImageDataGenerator(rotation_range=15,
#                                       rescale=1. / 255,
#                                       shear_range=0.1,
#                                       zoom_range=0.2,
#                                       horizontal_flip=True,
#                                       width_shift_range=0.1,
#                                       height_shift_range=0.1)
#
#     test_generator = train_datagen.flow_from_dataframe(train_df,
#                                                        "./dataset_all/", x_col='filename', y_col='category',
#                                                        target_size=Image_Size,
#                                                        class_mode='categorical',
#                                                        batch_size=batch_size)
#
#     if retrain:
#         epochs = 10
#         history = model.fit_generator(
#             train_generator,
#             epochs=epochs,
#             validation_data=validation_generator,
#             validation_steps=total_validate // batch_size,
#             steps_per_epoch=total_train // batch_size,
#             callbacks=callbacks
#         )
#
#         model.save("model1_10epoch.h5")
#     else:
#         model = tf.keras.models.load_model('model1_10epoch.h5')
#
#
#
#     test_filenames = os.listdir("./dataset_all/")
#     test_df = pd.DataFrame({
#         'filename': test_filenames
#     })
#     nb_samples = test_df.shape[0]
#
#     predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))
#
#     neki= np.argmax(predict, axis=-1)
#     test_df['category'] =neki
#
#     label_map = dict((v, k) for k, v in train_generator.class_indices.items())
#     test_df['category'] = test_df['category'].replace(label_map)
#
#     test_df['category'] = test_df['category'].replace({'m': 1, 'f': 0})
#
#     sample_test = test_df.head(18)
#     sample_test.head()
#     plt.figure(figsize=(12, 24))
#     for index, row in sample_test.iterrows():
#         filename = row['filename']
#         category = row['category']
#         img = load_img("./dataset_all" + filename, target_size=Image_Size)
#         plt.subplot(6, 3, index + 1)
#         plt.imshow(img)
#         plt.xlabel(filename + '(' + "{}".format(category) + ')')
#     plt.tight_layout()
#     plt.show()
#
#
#
# def main():
#     learn()
#     #test()
#
#
#
# if __name__ == '__main__':
#     main()


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////SECOND
# import numpy as np
# import pandas as pd
# import cv2
# from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from random import shuffle
# import tensorflow as tf
# import os
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, \
#     Dropout, Flatten, Dense, Activation, \
#     BatchNormalization
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.preprocessing import image
#
# Image_Width = 60
# Image_Height = 60
# Image_Size = (Image_Width, Image_Height)
# Image_Channels = 3
#
#
# def learn(retrain=False):
#     print("running")
#     filenames = os.listdir("./dataset_all")
#
#     categories = []
#     for f_name in filenames:
#         category = f_name.split('.')[0]
#         if category == 'm':
#             categories.append(1)
#         else:
#             categories.append(0)
#
#     df = pd.DataFrame({
#         'filename': filenames,
#         'category': categories
#     })
#
#     model = Sequential()
#
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Image_Width, Image_Height, Image_Channels)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.5))
#
#     model.add(Conv2D(16, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.5))
#
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.5))
#
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.5))
#
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#
#     model.add(Dense(2, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
#     model.summary()
#
#     df["category"] = df["category"].replace({0: 'f', 1: 'm'})
#     train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
#
#     train_df = train_df.reset_index(drop=True)
#     validate_df = validate_df.reset_index(drop=True)
#
#     train_datagen = ImageDataGenerator(rotation_range=15,
#                                        rescale=1. / 255,
#                                        shear_range=0.1,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                        width_shift_range=0.1,
#                                        height_shift_range=0.1)
#
#     train_generator = train_datagen.flow_from_dataframe(train_df,
#                                                         "./dataset_all/", x_col='filename', y_col='category',
#                                                         target_size=Image_Size,
#                                                         class_mode='categorical')
#
#     validation_datagen = ImageDataGenerator(rotation_range=15,
#                                             rescale=1. / 255,
#                                             shear_range=0.1,
#                                             zoom_range=0.2,
#                                             horizontal_flip=True,
#                                             width_shift_range=0.1,
#                                             height_shift_range=0.1)
#
#     validation_generator = validation_datagen.flow_from_dataframe(
#         validate_df,
#         "./dataset_all/",
#         x_col='filename',
#         y_col='category',
#         target_size=Image_Size,
#         class_mode='categorical',
#     )
#
#     if retrain:
#         epochs = 500
#         model.fit_generator(
#             train_generator,
#             epochs=epochs,
#             validation_data=validation_generator,
#         )
#
#         model.save("model2_500epoch.h5")
#     else:
#         model = tf.keras.models.load_model('model2_500epoch.h5')
#
#     for file in os.listdir("./dataset_all_test"):
#         img_pred = image.load_img("./dataset_all_test/" + file, target_size=(Image_Width, Image_Height))
#         img_pred = image.img_to_array(img_pred)
#         img_pred = np.expand_dims(img_pred, axis=0)
#         rslt = model.predict(img_pred)
#         print("./dataset_all_test/" + file + "   " + str(rslt))
#
#
# def main():
#     learn()
#     # test()
#
#
# if __name__ == '__main__':
#     main()
