Title: Inverse-Square Laws: A Physical Consequence of the Geometry of Space
Date: 2014-09-10 16:39
Tags: physics, geometry, kant, math
Summary: The ubiquitous inverse-square laws in physics are a necessary consequence of the three-dimensional nature of space. Barrow shows that Kant was the first to recognize the geometrical connection, although he got it backwards. I explain the geometrical reason for inverse-square laws and follow Kant's argument. Thanks, Kant. Thant.

John D. Barrow's [*The Constants of Nature*](https://www.goodreads.com/book/show/18926355-the-constants-of-nature?ac=1)[^barrow] mentions that Kant may have been the first to notice a connection between the dimensionality of space and physical inverse-square laws, such as [Newton's law of universal gravitation](http://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation):

\begin{equation}
F = G\frac{m_1 m_2}{r^2}
\end{equation}

This is interesting. Inverse-square laws are [everywhere](http://en.wikipedia.org/wiki/Inverse-square_law#Occurrences), and 3D space really does appear to be special[^ehrenfest]. Among other things, [stable orbits depend on it](http://en.wikipedia.org/wiki/Bertrand's_theorem). I first encountered, or at least bothered to think about, the geometric reason for inverse-square laws in an electrodynamics and antenna theory lecture on [free-space path loss (FSPL)](http://en.wikipedia.org/wiki/Free-space_path_loss), which says that the power of a received signal in free space also has a $r^{-2}$ distance dependence.

It is easy to understand this by picturing a point source radiating equally in all directions, i.e. spherically. Spherical symmetry is common in nature. At a distance $r$, the power is "spread out" over the surface of an imaginary sphere having area $4\pi r^2$. A similar argument can be made for gravity.

This image (by [Borb](http://en.wikipedia.org/wiki/File:Inverse_square_law.svg), licensed under CC-BY-SA) helps illustrate the idea. At each distance $d$ the same total effect is distributed over an area $d^2$:

<img style="float: center" src="/images/500px-Inverse_square_law.svg.png" alt="Point source acting over a spherical area">

This is clearly a consequence of the dimensionality of space as a sphere surface area scales with r^2 in 3-space. In general, an N-dimensional[^convention] sphere has surface area which scales with $r^{N-1}$. So in 4D space, other things being equal[^equal], we would experience inverse-cube gravitation, electromagnetism, acoustics, and so on.

Back to Kant. In his first published work (1747), "Thoughts on the true estimation of living forces"[^kant], Kant argues (Section 9) that space would not exist "if substances had no forces to act external to themselves". This leads him in Section 10 to ague that 3-dimensional space is a consequence of inverse-square gravity. Kant concludes that the inverse-square law is arbitrary and that "God could have chosen another, e.g., the inverse-cube relation" and that this would have resulted in a different sort of space. He goes on to suggest, (well before the development of anything like modern topology or differential geometry), that these spatial possibilities ought to be investigated.

It would seem that Kant got it backward, but Kant was like that.

Nonetheless, these insights (and others[^misc]) show that an important feature of many natural laws depends on "pure" geometrical truth moreso than physical reality; the inverse-square dependence is not due to an arbitrary constant exponent but the (admittedly, possibly arbitrary) dimensionality of space.

The Greeks would be proud.


[^barrow]: Barrow, John D, "New Dimensions," in *The Constants of Nature*. (New York: Random House, 2002), pp. 203-205

[^kant]: Kant, Immanuel, "Thoughts on the true estimation of living forces" in *Kant: Natural Science*, ed. Eric Watkins. 1st ed. (Cambridge: Cambridge University Press, 2012). pp. 26-28. Cambridge Books Online. Web. 10 September 2014. http://dx.doi.org/10.1017/CBO9781139014380.004

[^ehrenfest]: Ehrenfest, Paul. "In what way does it become manifest in the fundamental laws of physics that space has three dimensions." Proc. Royal Netherlands Acad. Arts Sci 20 (1917): 200-209.

[^convention]: By [geometer's conventions](http://mathworld.wolfram.com/Hypersphere.html). A topologist would call a 3D sphere a 2-sphere, as the surface has two dimensions.

[^equal]: To the extent that it is possible for anything else to remain the same, i.e. "probably not."

[^misc]: 2D and 3D spaces have many other special properties. See Ehrenfest (above) about rotation and wave propagation, Polya's work on the random walk on 2D, and Bertrand's Theorem.
