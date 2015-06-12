Title: Breaking reddit.com's CAPTCHA
Date: 2013-07-26 18:56
Tags: computer math, OpenCV, CAPTCHA, LSPC
Summary: CAPTCHAs are "a type of challenge-response test used in computing to determine whether or not the user is human." They are designed to be relatively easy for humans to solve, and difficult to automate. Some of them are very good, but the CAPTCHA system employed by reddit.com is, as of 2013-07-26, not state-of-the-art. Below, I attempt to solve this CAPTCHA automatically.

### Problem

[CAPTCHAs](http://en.wikipedia.org/wiki/CAPTCHA) are "a type of challenge-response test used in computing to determine whether or not the user is human." They are designed to be relatively easy for humans to solve, and difficult to automate. Some of them are very good, but the CAPTCHA system employed by [reddit.com](http://reddit.com/) is, as of 2013-07-26, not state-of-the-art. Below, I attempt to solve this CAPTCHA automatically.

A common approach to solving this kind of problem is to divide the problem into two parts: segmentation and recognition. First we attempt to divide the CAPTCHA into its single-character parts. Second, we use a classifier to match segments with labels. It has been shown [[^chellapilla]] that automatic classifiers perform well on single-letter images, and that segmentation is the more difficult problem.

![Example CAPTCHA (1)](/images/rmbc/ex1.png "Example CAPTCHA (1)")
![Example CAPTCHA (2)](/images/rmbc/ex2.png "Example CAPTCHA (2)")

### Method

#### Data

One thousand CAPTCHA images were collected. Each image is 8-bit 120x50 grayscale. All images have a six-character uppercase alphabetic solution, e.g. JVYKVC and TEMIWX, above. Half of the data was held out as a validation set, and hand-labelled with the correct solution by a human volunteer, Adrian C. Only 489 of these images were human-readable. We use the remaining five hundred images for segmentation.

#### Segmentation

[![Segmented CAPTCHA](/images/rmbc/segment4_t.png "Segmented CAPTCHA")](/images/rmbc/segment_4.png)
[![Segmented CAPTCHA](/images/rmbc/segment5_t.png "Segmented CAPTCHA")](/images/rmbc/segment_5.png)
[![Segmented CAPTCHA](/images/rmbc/segment6_t.png "Segmented CAPTCHA")](/images/rmbc/segment_6.png)
[![Segmented CAPTCHA](/images/rmbc/segment7_t.png "Segmented CAPTCHA")](/images/rmbc/segment_7.png)

Reddit's CAPTCHA algorithm employs a distorted grid which intercepts and joins each character in an attempt to prevent easy segmentation. However, these images are 8-bit rather than binary, and the anti-segmentation feature is rendered at a lower intensity than the letters themselves. This is the key weakness in this CAPTCHA algorithm.

By thresholding these images at a pixel value of 150 (chosen empirically), we remove much of this anti-segmentation grid, leaving noise. This also degrades the letters themselves at the edges, however, which leads to difficulty later on in the recognition phase. [Connected components](http://en.wikipedia.org/wiki/Connected-component_labeling) are then labelled, and components with fewer than twenty pixels (again chosen empirically) are discarded.

This process is shown visually in the four images above. Note that not every segmentation attempt is successful. See above a case above in which three letters remained joined by a particularly large remnant of a grid line. In some cases not pictured, letters are over-segmented, e.g. a 'W' is split into two 'V's, or an 'N' is incorrectly segmented at one of its vertices. Also, some noise components, especially at the borders, are still quite large and remain in the image. We deal with these in the next step.

#### Recognition

At this point we have done a reasonable, though not optimal job of segmenting these images. From the original five hundred training images, we now have 3,244 individual components, which I labelled by hand.

Some of these components are letters, but some are incorrectly-segmented multiple-letter sequences, half-letters, or pure noise. I trained a classifier, LSPC [[^sugiyama]] to distinguish four classes:

- Correctly-segmented letters
- Two characters incorrectly combined into one
- Three characters incorrectly combined into one
- Noise (non-letters, partial letters)

This classifier performs well, with a 3.58% misclassification rate (compared to a 16.77% misclassification rate if we simply always guess 'letter', the most probable case). In theory this classifier would allow us to attempt to fix the two- and three-character cases. In this example, I discard all non-letter components and proceed to the second classifier.

The second classifier attempts to categorize an input into one of twenty six classes (corresponding to the letters of the alphabet). After trying several versions of this classifier, I achieved the best misclassification error (10.14%), with a stacked autoencoder generating features for each image which were then fed to LSPC. In practice I actually used simple LSPC on raw pixel features (template matching), which achieved a misclassification error of 12.76%. I found that when analyzed in context of the entire system (segmentation+recognition), this classifier slightly outperforms the deep autoencoder.

This classifier assigns probability values to its guesses, so in the event that an image appears to have more than six segments, we take the most likely six. In the event that our segmentation process only finds five or fewer unique components, we abstain from guessing a solution (in practice, most interfaces have some feature to allow a user to request a new CAPTCHA, as they are occasionally unreadable for humans).

### Results

On the held-out validation set of 489 images, this system guessed only 5.73% (28) correctly. However, we knew a priori (because segmentation had failed), that we could not guess a number of images. If we abstain in those cases and request a new CAPTCHA, the success rate rises to an even 10%.

In a scenario where we have a limited number of incorrect attempts, but may request new CAPTCHAs without guessing, it is possible to achieve a success rate of 29% by only guessing on the 3% of the images we are most confident about.

Raw output of this test run is [available](/images/rmbc/log.txt).

### References

[^chellapilla]: K. Chellapilla, K. Larson, P. Simard, and M. Czerwinski, "Computers Beat Humans at Single Character Recognition in Reading Based Human Interaction Proofs (HIPs)," in *Proceedings of the Third Conference on E-Mail and AntiSpam*, 2005.
[^sugiyama]: M. Sugiyama. "Superfast-Trainable Multi-Class Probabilistic Classifier by Least-Squares Posterior Fitting". *IEICE Transactions on Information and Systems*, E93-D(10), pp. 2690–2701, 2010.
