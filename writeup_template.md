# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_test/test0.jpg "Traffic Sign 1"
[image5]: ./new_test/test2.jpg "Traffic Sign 2"
[image6]: ./new_test/test3.jpg "Traffic Sign 3"
[image7]: ./new_test/test5.jpg "Traffic Sign 4"
[image8]: ./new_test/test7.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because training the same traffic sign image with different color channel is not required,because both will give the same result,so training the model with grayscale is prefered, it will reduce the efforts required to train the model. For Example if we are training the model for stop sign in red and blue color channel will give the same result...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer 1        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image  					     	| 
| Convolution  3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| 
| Layer 2        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 14x14x6 grayscale image 						| 
| Convolution  3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |

| Layer 3        		|     Description	        					| 
| DropOut 				| 50% Keep										| 
| Fully connected		| Input = 400, Output = 120						| 
| Layer 4        		|     Description	        					| 
| DropOut 				| 50% Keep										| 
| Fully connected		| Input = 120, Output = 84 						| 
| Layer 5        		|     Description	        					|
| DropOut 				| 50% Keep										| 
| Fully connected		| Input = 84, Output = 10						| 
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a Adam Optimizer....
batch size=128
number of epochs=25
learning rate=0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.2%
* test set accuracy of 92.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    LeNet Architecture was the architecture that I used.Because it was suitable for my requirements.
* What were some problems with the initial architecture?
    Lack of data and lack of knowledge of other parameters. Adding a convolution layer help me reaching higher accuracy


* How was the architecture adjusted and why was it adjusted? 
    Convolution layers are added to the architecture to adjust it.
    and to avoid overfitting of moodel,dropouts are applied with 50% keep probability.
    
* Which parameters were tuned? How were they adjusted and why?
    Epoch: I set epochs at 25.Because i was able to get good probabitlity at this.
    The batch size : i kept it as default 128.
    The learning rate I have left at .001 as it was default good rate. I tried changing it as it mattered a little
    In Early steps The dropout probability mattered a lot but after a while. I set it to 50% and just left it. 
    
    
* What are some of the important design choices and why were they chosen?
    
    I learnt alot from this. as i tried differnet architecture to train my model with different values of all Hyperparameters.and also i played around with dropout rate to train my model accurately.
    

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| road Work      		| road Work   									| 
| No entry				| No entry										|
| Turn right ahead	    | Right-of-way at the next intersection			|
| Speed limit (50km/h)	| Speed limit (50km/h)      					|
| Stop      	      	| Stop					 				        |




The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 92.5% accurate...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 'Road Work' sign (probability of 0.99), and the image is predicted correctly. The top five soft max probabilities are

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| .99         			| Road Work  							| 
| .0001    				| General caution 						|
| .00002			    | Traffic signals				   		|
| .00	      			| Yield					 		    	|
| .00				    | Road narrows on the right      		|


For the second image of 'No Entry' Sign... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| No Entry  									| 
| .00     				| Speed limit (20km/h) 							|
| .00					| Speed limit (30km/h)							|
| .00	      			| Speed limit (50km/h)				     		|
| .00				    | Speed limit (60km/h)      					|

For the third image of 'Turn right ahead' sign...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Pedestrians                             		| 
| .001     				| General caution 								|
| .00					|  Right-of-way at the next intersection		|
| .00	      			| Speed limit (50km/h)				        	|
| .00				    | Dangerous curve to the right     	    		|

For the forth image 'Speed limit (50km/h)' sign... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (20km/h)  						| 
| .00     				| Speed limit (50km/h) 							|
| .00					| Speed limit (80km/h)                          |
| .00	      			| Wild animals crossing					 		|
| .00				    | Children crossing      				    	|

For the fifth image 'Stop' sign... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   								| 
| .00     				| No entry 									|
| .00					| Keep left									|
| .00	      			| Turn right ahead							|
| .00				    | Yield                  					|






### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


