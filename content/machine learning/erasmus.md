Title: A Localized Path-Finding Genetic Algorithm
Date: 2013-08-17 19:30
Tags: computer math, genetic algorithm, models, machine learning
Summary: I implement a genetic algorithm which attempts to find the lowest-cost path across some weighted map using only local information.

### Localized Path-Finding

Below I implement a genetic algorithm which attempts the following problem:

> Find a model which can, as an autonomous agent, traverse the lowest-cost path across some weighted map using only local information.

If this agent had prior knowledge of the entire map, determining the best path across it would be an objective (and well-studied) matter. In the following I attempt to train an agent to solve this problem with limited (local) information. These agents make decisions based only on four numbers: the relative heights of the terrain one "move" away in each of four directions (i.e., they can see no further than they can move).

On a uniformly-weighted space the lowest-cost path is simply the shortest, but here cost may be a proxy for difficult or rough terrain, or an abstraction. Below I will discuss the formulation and implications of the cost function used.

### Terrain and Sample Problem

[![Example Depth Map With Source, Destination, and Two Possible Paths](/images/erasmus/problem_ex_t.png "Example Depth Map With Source, Destination, and Two Possible Paths")](/images/erasmus/problem_ex.png)

The figure above shows one possible problem space. The grey levels in the image can be understood as a height map, with lighter pixels representing higher altitudes. The red circle is a starting position, and an agent must attempt to cross the vertical green line on the right border of the image. Shown in yellow and blue are two possible paths from source to destination.

A successful agent should strike a balance between the shortest path and one that involves the least amount of steep climbs or descents. It should attempt to follow altitude contours when possible, but occasionally climb (or descend) a hill for a significant shortcut.

Readers uninterested in the details of implementation may wish to skip to [results](#results) below.

### Cost Function

It is intuitive that the yellow, longer path is a worse solution than the blue, straightforward one. This notion can be made objective by computing a cost which is equal to the length of the path. I also discretize the problem: an agent only moves in quantum steps of length $\epsilon$, where $\epsilon$ is measured in pixels. This cost function is then proportional (or equal) to $N$, the number of steps required to complete the path:

\begin{equation}
C = N
\end{equation}

The shortest path (a straight line) is now always the best path, so this cost function must be augmented to take changes in altitude into account. The cost calculation becomes:

\begin{equation}
C = \sum\limits_{n=1}^N g_n^2 + \lambda N
\end{equation}

where<

\begin{equation}
g_n =
\begin{cases}
\Delta_n, & \text{if }\Delta_n \geq 0\\
-\frac{\Delta_n}{2}, & \text{if }\Delta_n < 0
\end{cases}
\end{equation}

for each step $g_n$ along the path. $\Delta_n$ is the change in height for that step. This cost is a function of both path length and the amount of climbing and descending one must do to complete the path. Note that cost scales with $g_n$ squared, meaning that very steep climbs and descents are penalized more strongly than shallow ones. Also, the cost for a descent is less than for a climb, but still positive (consider driving a vehicle down a steep, unpaved slope).

The parameter $\lambda$ controls the trade-off in cost between path length and altitude change: when $\lambda$ is very large, the path length component dominates and shorter paths will be selected at the cost of climbing. When $\lambda$ is very small, paths which minimize climbing will be favoured.

Finally, many agents never find a complete path. A (very large) fixed-cost penalty, $Q$, is added to their total costs.

### Agent Model

#### Model Input

I model an agent as a function which, at each step, decides which direction to move based on information about the relative height of its immediate surroundings. The input at each step, $\mathbf{x}$, is a vector of length $d$ of (an approximation of) the directional derivatives around the agent's current position:

\begin{equation}
x_i = \frac{h([x,y] + \epsilon\angle\theta_i)-h([x,y])}{\epsilon} \text{ for } i=1,...,d
\end{equation}

Here $h(x,y)$ represents the height at position $(x,y)$. The angles $\theta_i$ are spaced evenly along the unit circle. So the vector $\mathbf{x}$ is the difference in height between an agent's current position and $d$ surrounding locations which are $\epsilon$ units away. For a differentiable surface, the gradient of $h(x,y)$ is sufficient to describe it, but for this approximate model I have chosen $d=4$.

#### Model Parameters and Output

At each step the agent must decide a new direction in which to move. This should be an angle, $\phi$, between $\pi$ and $-\pi$. The model becomes a function $A$ of $\mathbf{x}$, parameterized by some unknown vector $\alpha$:

\begin{equation}
\hat{\phi} = A_\mathbf{\alpha}(\mathbf{x})
\end{equation}

The core of this model, $\mathbf{\alpha}^\top\mathbf{x}$, is a simple linear combination of the parameters, $\mathbf{\alpha}$, and the input vector $\mathbf{x}$. The output is then "squished" with a sigmoid function and normalized between $-\pi$ and $\pi$:

\begin{equation}
A_\mathbf{\alpha}(\mathbf{x}) = 2\pi\left(\frac{1}{1+exp(\mathbf{\alpha}^\top\mathbf{x})}-0.5\right)
\end{equation}

I show below that a model with only $d=4$ parameters is sufficient for reasonable performance as a path-finding agent. A genetic algorithm, below, is used to find near-optimal values for these four parameters.

### Genetic Algorithm

A [genetic algorithm](http://en.wikipedia.org/wiki/Genetic_algorithm) attempts to solve an optimization problem by mimicking evolution. Genetic algorithms are well-studied, so I will describe only my implementation for this problem. I represent each individual as a list of floating-point "genes" (in this case, each individual has four genes corresponding to the model parameters $\mathbf{\alpha}$.

- An initial population of $P=200$ individual agents is randomly generated. I initialize weights using a long-tailed t distribution. This encourages diversity in the gene pool by creating a few more relatively large weights (as opposed to a normal distribution) [[^montana]].
- Each agent is run and its total cost computed
- The best individual from the previous generation is carried over unchanged. This is referred to as elitism, and prevents the algorithm from throwing away the most fit individual through crossover or mutation [[^rudolph]]. It has also been shown to decrease convergence time [[^zitzler]].
- A new population is bred:
    - Two parents are selected with probability proportional to the reciprocal of their associated cost
    - The two parents are combined using two-point crossover, creating a child individual
    - These children are mutated by adding a small random offset to each gene with a 4% probability.
    - This is repeated until 90% of the new generation has been created
    - The remaining 10% of the new generation are initialized randomly

### Results

<a name="results"></a>

#### GA vs Control

In order to demonstrate that the genetic algorithm is productively solving this problem, I compare it to a control. The control algorithm keeps the best individual from the previous generation, as in the GA. Every other individual, however, is randomly generated. The following figures show that a genetic algorithm (first) outperforms a random search (second) of the parameter space. The cost function here has been tuned to discourage climbing in favour of longer, contour-following paths.


[![Path taken by best agent found by genetic algorithm](/images/erasmus/gavs_t.png "Path taken by best agent found by genetic algorithm")](/images/erasmus/gavs.png)
[![Path taken by best agent found by random selection](/images/erasmus/rndvs_t.png "Path taken by best agent found by random selection")](/images/erasmus/rndvs.png)

#### Evolution of Best Path

In the figures below, the best path for several different generations is evolved. The algorithm finds a global solution fairly quickly, and then optimizes certain local difficulties. The first image, below, shows the initial (random) best path, which does not reach the objective. After a few generations, a reasonable solution has been found but three problem areas (outlined in blue) remain. The next row of images focus on only the lower-right problem area. We see that the path, while initially confused, begins to straighten out after several generations.

[Text output from this experiment](images/erasmus/evolution.txt) is available. It is possible here to see how the percentage of the population which sucessfully completes any route grows as successful individuals are selected to reproduce.


[![Initial (random) solution](/images/erasmus/gen01_t.png "Initial (random) solution")](/images/erasmus/gen01_t.png)
[![Global (coarse) solution found after 11 generations. Local problem areas outlined in blue.](/images/erasmus/gen11_t.png "Global (coarse) solution found after 11 generations. Local problem areas outlined in blue.")](/images/erasmus/gen11_t.png)

[![Local problem area w/in global/coarse solution](/images/erasmus/gen11_p1_t.png "Local problem area w/in global/coarse solution")](/images/erasmus/gen11_p1_t.png)
[![Iteration upon problem area](/images/erasmus/gen15_p2_t.png "Iteration upon problem area")](/images/erasmus/gen15_p2_t.png)
[![Iteration upon problem area](/images/erasmus/gen22_p3_t.png "Iteration upon problem area")](/images/erasmus/gen22_p3_t.png)
[![Iteration upon problem area](/images/erasmus/gen36_p4_t.png "Iteration upon problem area")](/images/erasmus/gen36_p4_t.png)

#### Step Cost and Differentiability

Although the best path in the previous section eventually became somewhat smooth, its jagged nature can be improved upon. The jaggedness of that path can be attributed to two factors:

- The images in the previous section were generated using a low step cost $\lambda$. This low step cost does not serve well to discourage too-long paths.
- The image used as a map is discontinuous (i.e. not differentiable) at the transitions between white and black. In this case the agent has no information about the upcoming cliff until it hits it. It cannot choose an intermediate angle and must proceed in a zig/zag fashion.

By smoothing the image (by a 20x20 boxcar filter) and increasing the step cost, we can find a much better route through the image. From left to right:

- In the first figure below, the problem is demonstrated. The image is discontinuous, and $\lambda=0.001$.
- In the second figure, the image has been smoothed, but the step cost remains $\lambda=0.001$.
- Next, the step cost is increased to $\lambda=50$
- Next, the step cost is increased to $\lambda=1000$, and we see a relatively smooth path.
- Finally, we are shown a global view of the previous image (with $\lambda=1000$), showing the smoother path. Also note that some shortcuts have been taken.

[![Zig-zag path due to low step cost and discontinuous image](/images/erasmus/soln4_t.png "Zig-zag path due to low step cost and discontinuous image")](/images/erasmus/soln4_t.png)
[![Problem is ameliorated by smoothing the image](/images/erasmus/soln1_t.png "Problem is ameliorated by smoothing the image")](/images/erasmus/soln1_t.png)
[![Step cost is increased to lambda=50 resulting in a shorter path](/images/erasmus/soln2_t.png "Step cost is increased to lambda=50 resulting in a shorter path")](/images/erasmus/soln2_t.png)
[![Further increase in step cost (lambda=1000) results in a shorter, smoother path](/images/erasmus/soln3_t.png "Further increase in step cost (lambda=1000) results in a shorter, smoother path")](/images/erasmus/soln3_t.png)
[![Global view of previous image](/images/erasmus/soln3_global_t.png "Global view of previous image")](/images/erasmus/soln3_global_t.png)

#### Generalization

A simple experiment shows that these agents have some limited capacity for generalization. An agent was trained on a perlin noise depth map using an initial position near the centre of the image. It proceeded to its goal while attempting to maintain the same altitude throughout. The same agent was then run starting at the far left of the image (at a different altitude) and then found a reasonable path to its goal, again maintaining a relatively uniform altitude throughout.

[![Best route found during training](/images/erasmus/bestroute_trained_t.png "Best route found during training")](/images/erasmus/bestroute_trained_t.png)
[![Same agent released at a different starting position](/images/erasmus/bestroute_generalized_t.png "Same agent released at a different starting position")](/images/erasmus/bestroute_generalized_t.png)

This capacity for generalization has limits. In the following example, there is a conduit through the image with a fork. One fork is a route through, and the other is not. Because these agents work using only local knowledge, it is impossible to determine a priori which path is best (It is impossible for even a truly intelligent agent to choose the correct path given only the information at hand).

[![Trained agent on forked path finds a way through](/images/erasmus/gen01_t.png "Trained agent on forked path finds a way through")](/images/erasmus/gen01_t.png)
[![Same agent with the forks reversed fails to find the best route](/images/erasmus/gen01_t.png "Same agent with the forks reversed fails to find the best route")](/images/erasmus/gen01_t.png)

### Code

MATLAB code for this demo may be found on github: [iank/erasmus](https://github.com/iank/erasmus)

### References

[^rudolph]: G. Rudolph. Evolutionary search for minimal elements in partially ordered ﬁnite sets. In V. W. Porto, N. Saravanan, D. Waagen, and A. E. Eiben, editors, *Evolutionary Programming VII, Proceedings of the 7th Annual Conference on Evolutionary Programming*, pages 345–353. Springer, Berlin, 1998.

[^zitzler]: E. Zitzler, K. Deb, and L. Thiele. Comparison of Multiobjective Evolutionary Algorithms: Empirical Results. *Evolutionary Computation* 8.2 (2000): 173-195.

[^montana]: D. Montana and L. Davis. Training Feedforward Neural Networks Using Genetic Algorithms. *IJCAI*, vol. 89 (1989): 762-767
