#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.jpg "Model Visualization"
[image2]: ./examples/track1.jpg "Track1 Image"
[image3]: ./examples/center.jpg "Center Image"
[image4]: ./examples/left.jpg "Left Image"
[image5]: ./examples/right.jpg "Right Image"
[image6]: ./examples/origin_image.jpg "Normal Image"
[image7]: ./examples/flip_image.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 61-65) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 60). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 62,64,66,68,70).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drove around the first track in both a clock-wise and counter-clockwise direction.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a fully connected neural network, it drove poorly. Then I tried to use Lenet, I thought this model might be appropriate because convolution neural network worked pretty good at images.

In order to gauge how well the model was working, I run the simulator to see how well the car was driving around track one. The second  model drove much better than my first neural netwok, but still not good enough.

Then, I tried a more powerful mode. The model drove pretty well, but there were a few spots where the vehicle fell off the track, in order to improve the driving behavior in these cases, I added the images took from the left and right camera.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (the last cell) consisted of a few convolution neural network follow by a few fully connected layer. 
Here is the detail of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also used the right and left camera so that the vehicle could learn to recover from the left side and right sides of the road back to center. The three images below were taken from the following camera: center camera, left camera and right camera

![alt text][image3]
![alt text][image4]
![alt text][image5]

I also drove around the first track in both a clock-wise and counter-clockwise direction to prevent the data be biased towards left turns.

To augment the data sat, I also flipped images thinking that this would help, but the result was not good. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]



After the collection process, I had 7434 number of data points. I preprocessed this data by cropping images to get rid of the elements that might be distracting for the model,then I normalized the data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
