**Vehicle Detection**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_assets/early_result.png
[image2]: ./writeup_assets/early_result_2.png
[image3]: ./writeup_assets/false_positives.png
[image4]: ./writeup_assets/hog_vector.png
[image5]: ./writeup_assets/later_result.png
[image6]: ./writeup_assets/sample_frame.png
[image7]: ./writeup_assets/unbalanced_feature.png
[heatmap]: ./writeup_assets/heatmap.png
[video1]: ./project_video_output.mp4
[hog282]: ./writeup_assets/hog_282.png
[hog442]: ./writeup_assets/hog_442.png
[hog881]: ./writeup_assets/hog_881.png
[hog982]: ./writeup_assets/hog_982.png
[hog_o]: ./writeup_assets/hog_o.png


---
### Writeup

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features are extracted twice in the whole pipeline:

- for each training sample for the SVM classifier in `extract_features`
- globally for each video frame in `Processor.frame`. This increased the overall
  performance, because we do not call `get_hog_features` for each sliding window.


A sample plot of a HOG vector

![hog][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I started with the following HOG parameters:

```
orient=8
pix_per_cell=8
cell_per_block=1
```

Below I experimented with tuning them and decided to keep the initial settings.

| params | image |
| -- | -- |
| original subimage | ![hog][hog_o] |
| selected: **orient=8, pix_per_cell=8, cell_per_block=1:** | ![hog][hog881] |
| orient=2, pix_per_cell=8, cell_per_block=2: | ![hog][hog282] |
| orient=4, pix_per_cell=4, cell_per_block=2: | ![hog][hog442] |
| orient=9, pix_per_cell=8, cell_per_block=2: | ![hog][hog982] |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using scikit-learn `LinearSVC` class. This is the essential excerpt from the source code that includes labeling and scaling the dataset, splitting it into training and validation set and performing
training.

```
X = np.vstack((x_cars, x_not_cars)).astype(np.float64)
y = np.hstack((np.ones(len(x_cars)), np.zeros(len(x_not_cars))))

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

svc = LinearSVC()
svc.fit(X_train, y_train)
```

I have experimented with various feature vectors to get optimal classifier accuracy and settled
on YCrCb color space (all channels).


RGB - 3 channel
```
80.02 Seconds to extract HOG features
9.82 Seconds to train SVC
Accuracy: 0.9178
```

HSV - 3 channel
```
99.69 Seconds to extract HOG features
5.09 Seconds to train SVC
Accuracy: 0.962
```

HSV - only S
```
78.88 Seconds to extract HOG features
14.31 Seconds to train SVC
0.8727
```

YCrCb - 3 channel
```
194.51 Seconds to extract HOG features
39.24 Seconds to train SVC
0.9932
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use a range of six scales (from 1.1 to 3.5), but my laptop performance didn't allow me to
stick to it, as it would take about 2h to complete processing `project_video.mp4` (about 8s per frame).

Finally, I decided to use two scales:

- `ystart=384, ystop=512, scale=2.5`
- `ystart=384, ystop=660, scale=1.4`

These two passes gave optimal results and allowed me to acheive processing speed of just above 1 fps. Code for sliding window search could be found in `Processor.frame` and `find_cars` functions in the Jupyter notebook.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I started by experimenting with different hog parameters and color spaces but got a significant amount of
false positive.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Moreover I have rejected false positives by introducing heatmaps. Tuning their thresholds was essential to getting representative bounding boxes and elliminating false positives from the image.

Here in an example image before and after optimisation:

![false pos][image3]
![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I created a `Processor` class to store a python `deque` with last 10 frames of the video. New video frames
are being classified and converted to heatmaps using the `heatmap` function with a threshold of 3. These heatmaps are then cached in `Processor.heat_hist` deque and combined to create a final threshholded heatmap with actual vehicle detection.

Here's an example result showing the heatmap from a single frame (after thresholding). Result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frame of video:

![heatmap][heatmap]
![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. False positives - the pipeline is prone to false positives. It is difficult to fine an optimal parameter setup, as there's always a tradeoff between the stability and size of the correct bounding boxes, versus the number of random false positives on the video. I could devote some time in the future to improve this either using a smarter approach to sliding windows or a different classification method (a neural network?).

2. Performance - I had to reduce the number of scales from a range of 6 to just 2, so that the full video could
be processed in a reasonable time on my laptop. Even though I achieved almost 1 fps it is still way to slow to be considered ready for real time usage.

3. Limited search area - I can imagine the algorithm would fail on a hilly terrain, where the car is tilted forward or backward, because I have limited the effective viewport to just the middle-bottom range of the video.

All things considered, it was a challenging project, but there's still way to go to make it more robust. I would like to improve the false positive ellimination as well as work on improving the overall performance in the future.

