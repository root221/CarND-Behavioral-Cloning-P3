import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py

lines = []
with open('../data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    
images = []
measurements = []
for line in lines:
    filename = line[0].split('/')[-1]
    current_path = '../data1/IMG/' + filename
    image = mpimg.imread(current_path)
    images.append(image)
    measurement = float(line[3]) # line[3] type is str
    measurements.append(measurement)
        
# keras require numpy format
X_train = np.array(images)
y_train = np.array(measurements)
    
    
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D


car_images = []
steering_angles = []
for line in lines:
    images = []
    for i in range(3):
        filename = line[i].split('/')[-1]
        path =  '../data1/IMG/' + filename
        image = mpimg.imread(path)
        images.append(image)
                        
        steering_center = float(line[3])
       
        
    correction = 0.15
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    car_images.extend(images)
    steering_angles.extend([steering_center, steering_left, steering_right])

X_train = np.array(car_images)
y_train = np.array(steering_angles)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x:(x/255.0) - 0.5))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5)
model.save('model.h5')
