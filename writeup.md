#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./md_images/visualization.jpg "Visualization"
[image2]: ./test_images/1.png "Test Image 1"
[image3]: ./test_images/2.png "Test Image 2"
[image4]: ./test_images/3.png "Test Image 3"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images in each class:

![alt text][image1]

###Design and Test a Model Architecture

####1.

As a first step, I decided to convert the images to grayscale because reducing the amount of input data, training the model is significantly faster. The color of traffic signs should not be importent for classification. They are designed that even color blind people can identify them. And it works well according to YanLeCun Paper.

Here is an example of code to grayscale images:

```
# Convert to grayscale
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

X_train = X_train_gry
X_test = X_test_gry
```

As the next step, I normalized the image data before training for mathematical reasons. Normalized data can make the training faster and reduce the chance of getting stuck in local optima.

Then, I also added serveral techniques to the images of training dataset:

1. Random translate
2. Random Scaling
3. Random Warp
4. Random Brightness

This can increase the chance of predicting traffic signs under different conditions correctly.

I also shuffled the training dataset so that it will not obtain entire minibatches of highly correlated examples.

As the last step, I split the validation dataset off from the training dataset.


####2. 

My final model consisted of the following layers:

1. 5x5 convolution (32x32x1 in, 28x28x6 out) <-Layer 1
2. Tanh
3. 2x2 max pool (28x28x6 in, 14x14x6 out)
4. 5x5 convolution (14x14x6 in, 10x10x16 out) <-Layer 2
5. Tanh
6. 2x2 max pool (10x10x16 in, 5x5x16 out)
7. Flatten (5x5x16 in, 400 out)
8. 5x5 convolution (400 in, 120 out) <-Layer 3
9. Tanh
10. 5x5 convolution (120 in, 84 out) <-Layer 4
11. Tanh
12. Fully connected layer (84 in, 43 out) <-Layer 5
 


####3. 
To train the model, I used this model :

2. Epoch size : 30
3. Learning Rate : 0.0009
4. Batch size : 128
5. Mu = 0
6. Sigma = 0.01

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 93% 
* test set accuracy of 100%

I played with the epoch, learning rate and batch size several times, and ended up with current settings, which gives me a balanced efficiency and accuracy.

###Test a Model on New Images

####1. 
Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]


####2.

Here are the results of the prediction:

| Image											|     Prediction								| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection			| Right-of-way at the next intersection			| 
| Speed limit (60km/h)							| Speed limit (60km/h)							|
| Speed limit (30km/h)							| Speed limit (30km/h)							|
| Priority road									| Priority road									|
| Keep right									| Keep right									|
| Turn left ahead								| Turn left ahead								|
| General caution								| General caution								|
| Road work										| Road work										|



The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%.

####3. 

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 99.94%). The top five soft max probabilities were

[11 'Right-of-way at the next intersection']: 99.94%
[30 'Beware of ice/snow']: 0.03%
[34 'Turn left ahead']: 0.01%
[26 'Traffic signals']: 0.00%
[27 'Pedestrians']: 0.00%



For the second image : Speed limit (60km/h)

[3 'Speed limit (60km/h)']: 99.97%
[6 'End of speed limit (80km/h)']: 0.02%
[23 'Slippery road']: 0.00%
[5 'Speed limit (80km/h)']: 0.00%
[1 'Speed limit (30km/h)']: 0.00%

The rest are well predicted with probability of above 90%

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

Check out the output on the notebook.