r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
We can see from the graph we got that increasing ```k``` didn't lead to improved generalization for unseen data.
This is an example of the bias-variance tradeoff. Low values of ```k``` (like 1) yield in classifiers that are more commited to memory
and therefore overfit. Theses classifiers are very sensitive to noise - A noise might change the distance a bit and therefore changing the closet training point. High values can easily underfit, and maybe ```k``` will be larger than the dataset's size. Therefore, the middle values are the best.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Scaling $\Delta$ scales $W$, but the ratio between different $w_j$'s remains the same.
In SVM, scaling the weights with a positive constant doesn't affect the optimum.
Therefore, $\Delta$ sholudn't matter here.

"""

part3_q2 = r"""
First we can see there are 10 images, such that each image corresponds to a digit. So we can interpret each column in $W$ is a digit's weights.

The classification errors can be explained by this interpretation by the variance of the model weights. If a digit is written not clearly or rotated, then
the scores are different (because each pixel in $X$ now is changed), resulting in wrong classification.

This interpretation is similar to the KNN L2 minimization in sense of taking a maximum value in order to classify. The difference is over what we take the maximum value: in classifying using KNN we take argmax for only the K nearest neighbors from the training 
set, while in the linear classifier, we take argmax over the scores that correspond to the D classes. 

"""

part3_q3 = r"""
The learning rate is good, because we can see a clear convergence of the loss to a low value (below 0.002). If the learing rate was too low, then the converegence would be also too slow, and we couldn't see it in the graph. The graph of the loss would look like a 'streched' version of the current garph (i.e., more epochs are needed to a lower loss). If the learning rate was too high, then there is a possibility of not converging. This is because the algorithm would make dramatic changes to the weights at each iteration and miss the 'right' weights. The graph would look like a noisy channel with steep curves.

The model is slightly overfitted to the training set. In the graph we see the training set accuracy is higher than the validation set accuracy for any number of epochs (the difference between the two is not zero). Moreover, the difference appears to be constant after 5 epochs.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern is the horizontal line with 0 value (denote it by 0-line). The 0 values means there is no difference between the predicted value and the ground-truth value, i.e. a correct classification. Therefore, we are interested in points that are close to the 0-line.
We can see that the fitness of the trained model after CV is better than the top 5-features fitness. We can clearly see from the plot that the points are closer to the 0-line. Moreover, the variance is lower - there are almost no points that are too far from the ground truth (a distance of more than 5).

"""

part4_q2 = r"""
```logspace``` is used to save memory. This way, we need about 20 samples. If we used ```linspace``` we would need many samples for $\lambda \approx 10^{-3}$ to $\lambda \approx 10^2$.

The model was fitted 180 times. We fit for every possible triple ```(degree, lambda, fold)```. There are 3 degrees, 3 folds, 20 $\lambda values.

"""

# ==============
