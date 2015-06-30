Title: Feature Generation and Selection for Single Character Recognition
Date: 2015-06-29 16:17
Tags: python, MATLAB, computer math, OCR, feature selection
Status: Draft
Summary: TODO

- [Background: Single-character recognition](#background)
- [Dataset](#data)
- [Solutions](#solns)
    - [Template matching w/ LSPC (brute force benchmark)](#template)
    - [Hu moments (feature benchmark)](#hu)
    - [Receptors](#receptors)
- [Feature selection](#fs):
    - [Theoretical bound and PCA](#theory)
    - [Entropy (H(Y=1|X), H(X|Y=1))](#entropy)
    - [Redundancy (K-L divergence)](#redundancy)
    - [Greedy hill-climbing & pruning](#hill)
- [Results](#results)
- [Summary](#summary)
- [References/Notes](#refs)

<a name="background"></a>
### Background: Single-character recognition

In my [*Capitals*-playing project](/playing-capitals-with-opencv-and-python.html) I used the [Tesseract](https://en.wikipedia.org/wiki/Tesseract_(software)) OCR engine to read the letters in each tile after segmenting tiles with OpenCV.

Although my segmentation is consistent, presenting Tesseract with a single character from a single font on a white background, I noticed some performance problems. For example, the letter 'W' is never recognized correctly, and occasionally other letters are misclassified. Whether this is due to my using Tesseract incorrectly (i.e. possibly this could improve with [training](https://code.google.com/p/tesseract-ocr/wiki/TrainingTesseract3)), complications from using it in single character mode, or because Tesseract is [garbage](https://en.wikipedia.org/wiki/Waste), I'm not sure.

Also, Tesseract is a powerful OCR engine, but I only need single-character recognition. Invoking Tesseract for each tile in a screenshot takes a significant portion of the total runtime of [capitals-solver](https://github.com/iank/capitals-solver).

I decided to write my own classifier for the relatively trivial problem of classifying consistently-rendered single character tiles. This quickly became an exploration of [feature selection](https://en.wikipedia.org/wiki/Feature_selection) techniques. Later, I will explore the application of my model to more difficult problems, such as recognition of multiple fonts or handwritten letters.

All of the code used is available on [github/receptor-ocr](https://github.com/iank/receptor-ocr). I have tried to make note of when I used different revisions of the same code in this article.

I demonstrate a complete walkthrough of the most effictive method in the [README](http://TODO) and in [a subsequent post on the MNIST handwritten digit dataset](http://TODO).

<a name="data"></a>
### Dataset

I generated training/test data by segmenting [several screenshots](https://github.com/iank/receptor-ocr/tree/master/training_data) from [Capitals](https://itunes.apple.com/us/app/capitals-free-word-battle/id968456900?mt=8) using [gen_training_data.py](https://github.com/iank/receptor-ocr/blob/master/gen_training_data.py) and labeled them manually using [label_seg.py](https://github.com/iank/receptor-ocr/blob/master/label_seg.py). This [training data is available here](https://github.com/iank/receptor-ocr/tree/master/training_data), and is the same dataset I use throughout this article. A scaled example image is below.

![Example Q image](/images/receptors/E.png.cx_contour_12.png)

This is a $c=28$ class problem: A-Z, blank, and "capital" (an arbitrary icon specific to the game *Capitals*). I have a relatively small dataset, with $n=346$ examples. I have at least 4 examples of each class, except for 'J', of which there are only 3. The images are 500px by 500px binary images, or nominally $d=250,000$ dimensions. Throughout the article I use 75% of the data (259 instances) for training, and the remaining 25% (87 instances) for testing.

Note the small $n$ means that there is significant variability in performance, depending on how the training/test sets are (randomly) split. It's not improbable that there could be no 'J's present in the training set in one run, for example. Where error has varied significantly I have done several runs and presented a typical value.

Also note that since this is a multinomial problem, the 'chance' error rate is not 50%, but approximately 90%, obtained by always guessing the most frequent class, 'A'.

<a name="solns"></a>
### Solutions 

<a name="template"></a>
#### Template matching w/ LSPC (brute force benchmark)

*Note: Throughout this article I use LSPC[[^lspc]]. It is a kernel-based nonlinear classifier, approximately as accurate as SVM or KLR, but blindingly fast (it has a closed form solution), allowing me to evaluate hundreds of models in the time it would take to train a single KLR. It is also natively multi-class and produces probabilistic outputs.*

A naïve approach is to encode each image as a vector of pixel values and use these as inputs to a nonlinear classifier, such as a neural network, support vector machine, or [KLR](http://web.stanford.edu/~hastie/Papers/svmtalk.pdf). This is essentially template matching, and I'll refer to it as such here.

Template matching can be effective when images are consistent within classes (i.e. all "f"s look the same). On more difficult problems, template matching can be effective with some application-specific preprocessing, such as deskewing[[^deskewing]], and other normalization.

Encoding entire images as vectors can lead to dimensionality problems, even after reducing image resolution. Also, template matching models cannot make more "perceptual" decisions ("two straight lines at right angles, plus a curving part might be a five"). Part of the allure of deep neural networks is that their heirarchical structure may be enabling this type of decision making.

Since all character examples in my application come from a single font and are consistently oriented, template matching is a good benchmark.

Using [template_match.py](https://github.com/iank/receptor-ocr/blob/master/template_match.py) I generated the $n=346$, $d=250,000$ vectors and classified using LSPC by passing the resulting 165 megabyte CSV to [template.m](https://github.com/iank/receptor-ocr/blob/master/writeup/template.m).

The resulting data is unwieldy, and clearly redundant (as a back-of-the-envelope measure of information content, note that it gzips to 1.2M)

[![raw image data](/images/receptors/template_data_s.png)](/images/receptors/template_data.png)

Test error obtained with LSPC for template matching was 31.03%, significantly better than chance but not useful.

__Test error for template matching: 31.03%__

<a name="hu"></a>
#### Hu moments (feature benchmark)
 
[Hu invariant moments](https://en.wikipedia.org/wiki/Image_moment#Rotation_invariant_moments) are image moments which are invariant under translation, scaling, and rotation. They are not meant as pattern recognition features, however I use them here to provide another benchmark. This can be thought of as a slightly more sophisticated, image processing-specific version of classifying data based on their summary statistics (mean, variance, etc).

This isn't completely off the rails, for example in this representation blank spaces are all zeros, and it's easy to imagine that some more patterns will be separable.

Using [hu_moments.py](https://github.com/iank/receptor-ocr/blob/master/hu_moments.py) I generated the $n=346$, $d=7$ vectors and classified using the same MATLAB code as above.

Test error obtained with LSPC for Hu features was 10.34%, showing even rudimentary features can outperform template matching. Less is more. Small Data! [^smalldata]

__Test error for Hu features: 10.34%__

<a name="receptors"></a>
#### Receptors

With better feature design, we can generate smaller models, achieve better classification, and possibly gain insight into the data by producing more *explainable* models. The disadvantage to this approach over a more general method is that the features produced are often domain-specific, design can be labor-intensive, and approaches can be non-obvious or unknown in some domains.

Example features for character recognition could be:

- counting of connected components
- drawing regularly-spaced horizontal and vertical lines over the image and counting intersections for each
- statistics gleaned from approximating edges as lower-degree polygons
- etc

As an interesting compromise between manual feature design and automatic feature extraction, I found this [codeproject post by Andrew Kirillov](http://www.codeproject.com/Articles/11285/Neural-Network-OCR), who uses "receptors" (scroll to "Another Approach")[^receptors].

The idea is to project the same set of small line segments on each image, and generate a vector of receptor activation (crossing the letter / not crossing the letter) for each image.

In the codeproject post, the author is using a neural network, and training time can be greatly improved by reducing the number of features. He uses empirical estimates of entropy to attempt to select the most useful features.

I construct receptors by generating a set of midpoints, lengths, and angles. The midpoint positions are normalized so the image centroid is $(0.5,0.5)$, and distances (length, offset from centroid) are normalized by the image diagonal. So these features should not depend on scale or translation (although significant variation in whitespace padding could break things).

Midpoints have a Gaussian distribution $N(\mu = [0.5, 0.5], \sigma^2 = 0.2)$, and lengths are Rayleigh distributed ($\sigma = 0.08$). Angles are uniform between $0$ and $2\pi$.

At first, I tried to improve upon the receptor model by making receptor activation a real number (the average pixel intensity across the receptor) rather than a binary activation.

With [gen_receptors.py](https://github.com/iank/receptor-ocr/blob/master/gen_receptors.py) I generated 2500 receptors, and produced a CSV using [gen_training_csv.py](https://github.com/iank/receptor-ocr/blob/master/gen_training_csv.py).

Test error obtained with LSPC for 2500 real-valued receptors was 22%. By clamping receptor values to binary 0/1, I obtained perfect classification.

Here, $n=2500$ receptors is more than enough to perfectly classify my dataset, and training time with LSPC is a fraction of a second, making feature selection unnecessary. However, requiring so many receptors for such an easy classification task seems inelegant, so I discuss feature selection below.

__Test error for 2500 binary receptors: 0.00%__

<a name="fs"></a>
### Feature selection
<a name="theory"></a>
#### Theoretical bound and PCA

While 2500 binary features perfectly separates the data, $c=28$ classes should be separable with $\lceil log_2(28) \rceil = 5$ binary features[^log2], if five features can be found which:

- are consistent within-class (i.e. feature is *always* or *never* on whenever it is shown an example of an 'S')
- usefully separate the space (i.e. a feature that is always on for A-M and always off for N-Z. a second feature is on for A, C, E, ... and off for B, D, F, ...)

These qualities can be measured statistically with [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), but I first encountered this as a child playing [Guess Who?](https://boardgamegeek.com/boardgame/4143/guess-who): You attempt to select an individual from an array of faces by asking your opponent yes/no questions. "Is the person I am looking for female?" is a good first question to ask, because it splits the space evenly by eliminating a gender[^gender]. "Does the person have green eyeglasses" is highly-specific and could pay off, but it's more likely to eliminate only one or two faces, so it is not a great first question. I'll discuss this more in the next section.

Although I know that only five binary features could separate 28 classes, I don't yet know that these features can be modeled with receptors (I have strong evidence that such features *exist*, however, since I have separated the space using 2500 receptors).

More evidence that I can use significantly fewer than 2500 receptors: I used [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), which I have written about [previously](http://blog.iank.org/pca-on-x-plane-images.html), on the receptor activation data. Using PCA I can project the data along the first $k$ principal components, which is like re-shaping the data along a new set of axes which are uncorrelated.

It's in part a way to get at the underlying dimensionality of the data: a $d \times d$ matrix will have $d$ principal components, but only a few may be large. In my test with 4841 receptors only about 10 principal componennts were significantly larger than zero, suggesting that a low-dimensional representation will be sufficient to represent most of the information.

I used [rocr_pca.m](https://github.com/iank/receptor-ocr/blob/master/writeup/rocr_pca.m) to do PCA on the receptor activation data. Using the first $k$ principal components[^pcacont]:

<table>
<tr><th>k</th><th>LSPC test error</th></tr>
<tr><td>1</td><td>24.14%</td></tr>
<tr><td>2</td><td>5.75%</td></tr>
<tr><td>3</td><td>1.15%</td></tr>
<tr><td>4</td><td>0.00%</td></tr>
<tr><td>5</td><td>0.00%</td></tr>
</table>

(Note that fewer than five principal components were needed. This is because the projection along these components is real-valued, not binary).

With, for example, 5 principal components, I still need to compute the activation for all 4841 receptors before projecting in order to classify an image[^pca]. But this experiment shows that the underlying information is not nearly 4841-dimensional and far fewer receptors should be needed. Selecting useful receptors is explored in the following section.

__Test error for 4 principal components (from 4841 binary receptors): 0.00%__

<a name="entropy"></a>
#### Entropy

[Andrew Kirillov](http://www.codeproject.com/Articles/11285/Neural-Network-OCR) uses entropy to do feature selection. I reimplemented this approach at first, and I'll summarize it here. The upshot is that it can reduce the number of features required, but not as dramatically as hill-climbing (below).

##### Probability

I know that only a few receptors should be required. I generate a field of 5,000 receptors, which should be large enough to contain a few useful features. Receptor activation (binary on/off)  I model with a random variable $Y_k \in \\{0,1\\}$ (where k represents the particular receptor in question, 1 to 5,000). A random variable $X \in \\{A, B, C, ...\\}$ models which class a given image is.

Some quantities, which I empirically estimate from my training images:

* $p(X)$: class frequency distribution, e.g. $p(A) = 0.1069$, $p(W) = 0.0173$
* $p(Y_k=1)$: probability that receptor $k$ is on (across all images)
* $p(Y_k=1|X=x)$: probability that receptor $k$ is on, **given** that the image is a specific letter $x$

For example, if receptor 903 is always on when the letter is 'Q', then $p(Y_{903}=1|X=q) = 1$. This means that all 'Q's trigger receptor 903, which is good, but it does not mean that receptor 903 is a good indication that the letter is a 'Q'. 

Using Bayes rule, I can compute $p(X=x|Y_k=1)$, the probability that an image is a specific letter given that receptor $k$ is on:

\begin{equation}
p(X=x|Y=1) = \frac{p(Y=1|X=x) \cdot p(X=x)}{P(Y=1)}
\end{equation}

Receptor 903 may light up for W's and A's also (or for every letter! In other words, $p(X=q|Y_{903}=1)$ may still be small).

##### H(Y|X)

So two quantities are interesting. We want receptors which are consistent within their classes. We can measure this by the entropy $H(Y_k|X)$. This is the average of the conditional entropy $H(Y_k|X=x)$ across all X. For example, if $H(Y_{k}|X=q)$ is large, then there is uncertainty about whether receptor k will turn on for a 'Q'. If it is zero, then receptor k is either *always* on or *always* off when presented with a 'Q'.

##### H(X|Y)

More interesting is how well a receptor determines class. This is $H(X|Y_k=1)$, which captures the uncertainty in $X$ given that $Y_k$ is on. It is minimized by certainty (if $p(X=x|Y=1) = 1$ for some $x$), and maximized ($-log_2(1/C)$) by a uniform distribution, indicating complete uncertanty.[^goodquestion]

I select receptors which are consistent and divide the space by selecting small $H(Y_k|X)$ and large $H(X|Y_k=1)$. This is done by assigning to each receptor a 'usefulness' score defined by the product:

\begin{equation}
H(X|Y_k=1) \cdot (1 - H(Y_k|X)),
\end{equation}

and selecting the receptors with the $N$ highest values. By classifying using LSPC with the first $N$ receptors, I plot test error vs $N$, showing how many receptors are required for successful classification (from an initial set of 2500):

![err_vs_n.png](/images/receptors/err_vs_n.png)

So only about 600 receptors are required for perfect classification which is an improvement, but this image shows that the first nearly 200 receptors are completely useless! 

I also tried selecting for receptor specificity, i.e. recomputing usefulness as:

\begin{equation}
(-log_2(1/C) - H(X|Y_k=1)) \cdot (1 - H(Y_k|X)),
\end{equation}

![err_vs_n_2.png](/images/receptors/err_vs_n_2.png)

Note the larger horizontal axis here. With this method, the first few receptors were more immediately useful, but it took longer to converge. Also, this is from an initial set of 5000. This is evidence that there is a balance between initial set size (larger = more likely to generate useful receptors) and redundancy.

<!-- This method used revisions 2fb264a9a0e759eaa06e0c0a9cc263c655f78f17 and 3706c08a1d067cf13a3d2efd3c4317fc76071e44 of gen_receptors.py TODO -->

Both methods are a clear improvement over 2,500 or 5,000, but it seems too many. This is addressed in the next section.

__Test error for first 600 features selected using entropy: 0.00%__

<a name="redundancy"></a>
#### Redundancy (K-L divergence)

The entropy-based selection described above has a certain information-theoretic appeal, but it does not look for receptors which split the class space *differently*. A receptor which separates A-N from M-Z has high 'usefulness' as calculated above, but so do 200 receptors which separate A-N from M-Z, and these do not add information beyond the first. In the image below, the first 1,000 or so receptors (shown in green) very specifically identify a capital. Only after 1,000 do we get receptors which activate for other letters, such as 'A' (red).

![c1000.png](/images/receptors/c1000.png)

I attempted to address this by augmenting the usefulness score with a measure of redundancy[^mrmr]. The [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is a nonnegative measure of difference between probability distributions, closely related to mutual information and entropy. $D_{KL}(P||Q)$ is not symmetric, but $D_{KL}(P||Q) + D_{KL}(Q||P)$ is. So I use the following algorithm to score receptors:

- Compute usefulness for all receptors as above
- Until no receptors are remaining:
    - Add most useful receptor to an ordered set $S$
    - Recompute usefulness for all receptors, multiplying by the average symmetric K-L divergence from receptors in $S$
- Take the first $N$ receptors from $S$, these are the $N$ most 'useful' receptors

<!-- TODO revision e5be113fbe1dccdad865c78ccdf5cfef10eae127 of gen_receptors.py -->

The result of this approach, from an initial set of 5000, is shown below (here I am minimizing $H(X|Y)$, as in the previous plot).

![err_vs_n_3.png](/images/receptors/err_vs_n_3.png)

Initially, this is a clear improvement over the last, but it has not succeeded in eliminating many features in the end. This leads me to abandon the information-theoretic approach entirely in favour of a simple method discussed in the next section.

<a name="hill"></a>
#### Greedy hill-climbing & pruning

<!-- TODO: rocr_hill.m -->

A simple feature selection strategy is greedy hill-climbing. Start with an empty set of features, $S$, and a set of remaining features, $R$.

- Until $R$ is empty **or** the test error has not improved in a few iterations:
    - For every feature $k \in R$, train a classifier with features $S + k$.
    - Add whichever feature decreased test error the most to $S$
    - Remove it from $R$

This involves training an astonishing number of classifiers, like a [triangular number](http://mathworld.wolfram.com/TriangularNumber.html) $T_n$ for $n$ initial features. Fortunately this is possible in a reasonable period of time with LSPC, and even fast if I add the five best features at each iteration instead of proceeding one at a time.

The "greedy" aspect of this algorithm is that it never un-selects a feature. It is descending[^hillclimb] an objective function (test error) by taking the steepest immediate step at every iteration. A complete search would be the power set of features, $2^{5000}$ of them, but that is too many and the greedy hill climb works well enough.

Since I added five features at a time, hill climbing is followed by a pruning step, which is much the same but in reverse. One feature is removed at a time.

<a name="results"></a>
### Results

In a test with 5000 initial receptors, 45 were added and then pruned to 20 while maintaining perfect classification. These receptors are shown below on a blank image and two letters:

![field_labels.png](/images/receptors/field_labels.png)
![field_s.png](/images/receptors/field_s.png)
![field_w.png](/images/receptors/field_w.png)

The features are shown below, along with their entropies and probabilities of activation $p(Y_k=1|X=x)$. This helps illustrate how each feature breaks up the space:

<table style="border-collapse: collapse;">
<tr><td style="padding: 0; margin: 0; background-color:#00FF00">On (&gt; 90%)</td>
    <td style="padding: 0; margin: 0; background-color:#ADD8E6">Mid (15% - 90%)</td>
    <td style="padding: 0; margin: 0; background-color:#FFE3EB">Low (1% - 15%)</td>
    <td style="padding: 0; margin: 0; background-color:#FFFFFF">Off (&lt; 1%)</td>
</tr>
</table>

<table style="border-collapse: collapse;">
<tr><th>#</th><th>Hyx</th><th>Hxy</th><th>_</th><th>1</th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th><th>F</th><th>G</th><th>H</th><th>I</th><th>J</th><th>K</th><th>L</th><th>M</th><th>N</th><th>O</th><th>P</th><th>Q</th><th>R</th><th>S</th><th>T</th><th>U</th><th>V</th><th>W</th><th>X</th><th>Y</th><th>Z</th></tr>
<tr><td>4621</td><td>0.06</td><td style="border-right: 1px solid">4.50</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td></tr>
<tr><td>2133</td><td>0.08</td><td style="border-right: 1px solid">3.92</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td></tr>
<tr><td>2598</td><td>0.04</td><td style="border-right: 1px solid">3.81</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td></tr>
<tr><td>3035</td><td>0.11</td><td style="border-right: 1px solid">3.78</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td></tr>
<tr><td>2404</td><td>0.10</td><td style="border-right: 1px solid">3.65</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>1571</td><td>0.13</td><td style="border-right: 1px solid">3.33</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>2186</td><td>0.07</td><td style="border-right: 1px solid">3.31</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>2151</td><td>0.15</td><td style="border-right: 1px solid">3.28</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td></tr>
<tr><td>1460</td><td>0.07</td><td style="border-right: 1px solid">3.24</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>1777</td><td>0.03</td><td style="border-right: 1px solid">3.09</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>1195</td><td>0.09</td><td style="border-right: 1px solid">2.98</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td></tr>
<tr><td>1645</td><td>0.19</td><td style="border-right: 1px solid">2.85</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td></tr>
<tr><td>1202</td><td>0.11</td><td style="border-right: 1px solid">2.40</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>1330</td><td>0.14</td><td style="border-right: 1px solid">1.84</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>154</td><td>0.09</td><td style="border-right: 1px solid">1.46</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>1145</td><td>0.05</td><td style="border-right: 1px solid">1.06</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#ADD8E6">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>177</td><td>0.01</td><td style="border-right: 1px solid">0.00</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>392</td><td>0.01</td><td style="border-right: 1px solid">0.00</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>445</td><td>0.01</td><td style="border-right: 1px solid">0.00</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFE3EB">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
<tr><td>315</td><td>0.01</td><td style="border-right: 1px solid">-0.00</td><td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#00FF00">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td> <td style="padding: 0; margin: 0; background-color:#FFFFFF">&nbsp;</td></tr>
</table>

<a name="summary"></a>
### Summary

Here is a summary of the techniques discussed in this article. In all cases (except chance), LSPC is used to classify the generated/selected features.

<table>
<tr><th>Method</th><th>Features</th><th>Error</th></tr>
<tr><td>Chance</td><td>0</td><td>~90%</td></tr>
<tr><td>Template matching</td><td>250000</td><td>31.03%</td></tr>
<tr><td>Hu moments</td><td>7</td><td>10.34%</td></tr>
<tr><td>PCA (continuous activation)</td><td>8 principal components (4841 underlying receptors)</td><td>0%</td></tr>
<tr><td>PCA (binary activation)</td><td>4 principal (4841 underlying receptors)</td><td>0%</td></tr>
<tr><td>Receptors (continuous activation)</td><td>2500</td><td>22%</td></tr>
<tr><td>Receptors (binary activation)</td><td>2500</td><td>0%</td></tr>
<tr><td>Entropy selection (max HXY)</td><td>~600 (initially 2500)</td><td>0%</td></tr>
<tr><td>Entropy selection (min HXY)</td><td>~1700 (initially 5000)</td><td>0%</td></tr>
<tr><td>Entropy (min HXY) + K-L divergence</td><td>~1500 (initially 5000)</td><td>0%</td></tr>
<tr><td>Greedy hill-climbing &amp; pruning</td><td>20 (initially 5000)</td><td>0%</td></tr>
</table>

<a name="refs"></a>
### References/Notes

[^lspc]: [Sugiyama, Masashi. "Superfast-trainable multi-class probabilistic classifier by least-squares posterior fitting." *IEICE Transactions on Information and Systems* 93.10 (2010): 2690-2701.](http://www.ms.k.u-tokyo.ac.jp/2010/LSPC.pdf)

[^deskewing]: Or 'deslanting', discussed in Teow, Loo-Nin, and Kia-Fock Loe. "Robust vision-based features and classification schemes for off-line handwritten digit recognition." *Pattern Recognition* 35.11 (2002): 2355-2364.

[^smalldata]: This illustrates an inefficiency in the model: the template matching dataset contained more information, extracting the Hu features did not add any information. By rearranging information and throwing away redundant data, I was able to improve performance here. It is theoretically possible that a sufficiently complex neural network could perform optimally (here, perfect classification is possible) on the raw data. In some cases, this is impractical and some preprocessing can go a long way. On the other hand, especially in image recognition tasks, [DBNs](https://en.wikipedia.org/wiki/Deep_belief_network) and other deep neural models have shown remarkable results and can be used to generate high-level features automatically. It's possible that this kind of model can make manual feature design unnecessary, and enable feature design in spaces we do not understand.

[^log2]: One feature can split 28 classes into two groups of 14. Further independent features could split those into four groups of 7, then four groups of 3 and four groups of 4, and so on. [28] -> [14 14] -> [7 7 7 7] -> [3 3 3 3 4 4 4 4] -> [2 2 2 2 2 2 2 2 2 2 2 2 1 1 1] -> [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]. So 5 features are needed.

[^gender]: The game came out in 1979, so this is binary.

[^pcacont]: With continuous rather than binary receptor activations, PCA+LSPC required 8 principal components.

[^pca]: For LSPC, this means PCA is no real advantage. It is cheap enough to classify $d=2500$ vectors. For models which depend heavily on dimension, like multi-layer perceptrons, PCA will be a big speedup.

[^goodquestion]: Maximizing this value, then, would seem counterintuitive. However, receptors which evenly split the class space also have high entropy. I also tried minimizing H(X|Y), which is more akin to asking "does the subject wear green eyeglasses?" than "does the subject have facial hair?", but small values here at least ensure there is some certainty. Both approach suffer from the redundancy issue I discuss in the next section.

[^mrmr]: This combined entropy/redundancy approach is vaguely reminiscent of more mature techniques like [mRMR](https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_.28mRMR.29_feature_selection).

[^hillclimb]: I guess nobody calls it "valley-descending"

[^receptors]: It's worth noting the similarities to template matching here. These features are not invariant to rotation or skew, but are somewhat more flexible than templates because activation may take place anywhere on an arbitrary line segment.
