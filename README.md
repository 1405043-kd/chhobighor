# Underwater Image Enhancement

cv.py file contains the coding portion. Sample images are also provided. To run the cv.py in your local machine give an underwater image path as input at 6th line.  

# this is an almost similar implementation of this paper, https://ieeexplore.ieee.org/document/8058463/

The whole process contains roughly 5 major steps.

In step 1, the underwater image is white balanced by compensating red and blue channels.

In step 2, two images are generated from this white balanced version. One image is unsharp masked and another is gamma adjusted. Thus
one generated image ensures details of images and another adjusts contrasts.

In step 3, three weights are generated from each of two inputs. First one is Laplacian weight, second one is saliency weight and the 
third one is saturation weight. To generate saliency weight, according to the paper, 

##R. Achantay, S. Hemamiz, F. Estraday, and S. Susstrunk, “Frequencytuned salient region detection,” in Proc. IEEE CVPR, Jun. 2009

is used. However, here the naive equation described in Achatay et al. has been implemented. 

In step 4, Aggregated weight of these three weights is generated for each of the two images from step 2. Then average weight is generated for 
each of the two aggregated weights.

In step 5, according to the paper multi-scale pyramid based fusion is used. However, to reduce complexity, here a weighted sum of the inputs and generated 
weights has been implemented. 
