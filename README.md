# Image-Segmentation

# Dataset
a. We will use Berkeley Segmentation Benchmark

b. The data is available at the following link.
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bs
ds500.tgz.

c. The dataset has 500 images. The test set is 200 images only. We will report our
results on the first 50 images of the test set only.

# Segmentation using K-means
Every image pixel is a feature vector of 3-dimension {R,G,B}. We will use this feature
representation to do the segmentation.

a. We will change the K of the K-means algorithm between {3,5,7,9,11} clusters.
You will produce different segmentations and save them as colored images. Every
color represents a certain group (cluster) of pixels.

b. We will evaluate the result segmentation using F-measure , Conditional Entropy .
for image I with M available ground-truth segmentations. For a clustering of
K-clusters you will report your measures M times and the average of the M trials
as well. Report average per dataset as well.

