Title: Autorouting PCBs
Date: 2013-08-11 18:30
Tags: computer math, EDA, routing
Summary: Routing, in electronic design, is a difficult and often intractible problem. Usually it is done with at least some human intervention. Most Electronic Design Automation packages include sophisticated software tools to allow the routing process to be machine-assisted or, in some cases, entirely automatic. Below I investigate some of the simplest methods for solving this problem, and demonstrate a working (albeit trivial) example of an automated approach.

### Problem

[Routing](http://en.wikipedia.org/wiki/Routing_(electronic_design_automation)), in electronic design, is a difficult and often intractible problem. Usually it is done with at least some human intervention. Most Electronic Design Automation packages include sophisticated software tools to allow the routing process to be machine-assisted or, in some cases, entirely automatic. Below I investigate some of the simplest methods for solving this problem, and demonstrate a working (albeit trivial) example of an automated approach.

The problem of routing deals with 'nets', which are lists of pins or pads which must be connected by some conductor. Also, different nets should not be electrically connected. This class of error is called a 'short'. Shorts with other nets can be avoided when routing a net by:

- Simply routing around any existing routes which are in the way (as in Tron)
- Using multiple layers to allow routes to avoid each other 'vertically'. This requires placing [vias](http://en.wikipedia.org/wiki/Via_(electronics)), which introduces additional cost.
- As a last resort, a jumper wire can be introduced in order to complete a route.

Other design rules must be obeyed. In the case of printed circuit boards, fabrication processes have minimum trace widths, spacing tolerances, and so on. Wire length must be minimized and, depending on the domain, there are other electrical constraints to be considered.

### Method

Several simple algorithms are described in [[^zhou]] below. The Lee Algorithm is straightforward, and can be used to route nets with multiple nodes. Routing can also take place on a weighted grid. Instead of minimizing wire length (as if all weights were unity), we then attempt to find paths with minimum cost. In future problems I assign weights in order to discourage (but still allow, as a first pass) errors such as shorts or design rule violations. In the demonstration below, however, all weights are either unity or infinity (blocked cells, e.g. pads and already-routed tracks).

This approach finds many possible paths from some source[s] to a destination node by wave propagation, then picks the lowest-cost path by tracing backwards. It may be useful to skip to the [visualization](#vis) below before proceeding.

#### Routing a single net

Setting aside for a moment the problem of choosing the order in which nets should be routed, we consider the problem of finding the best (or lowest-cost) route to a grid cell $T$ from one or many cells $\vec{S}$ (In the case of a three-node net in which two nodes have already been routed, we may connect the remaining node to any point in $\vec{S}$).

This approach finds many possible paths to $T$ from $\vec{S}$ by propagating a wave from $\vec{S}$. We begin by considering the neighbors of $\vec{S}$, and marking them with the cost to reach each cell.

For each $S_i$ in $\vec{S}$:

- Update cost to reach each neighbor, $N_j$ of $S_i$: $cost(N_j) = cost(S_i) + weight(N_j)$
- (Some $N_j$ may already be marked. If so, only update if the new cost is less than the current)
- Stop once we reach $T$. In the case of a weighted grid, we should proceed until the route surronds $T$, in case a lower-cost path can be realized by approaching from a different direction.

An example of a propagated wave from $S$ to $T$ (labelled) is shown below. The deep red inclusions in the wave are cells with infinite weight (obstacles or pads belonging to different nets).

[![Wave from S to T showing many possible paths](/images/waveprop_t.png)](/images/waveprop.png)

Now that each cell has been marked with a cost, the total cost for each discovered path can be computed by the sum of the cost for each cell along the way. The lowest-cost path can be discovered more simply, by backtracing a path from $T$ to any $S$. (Simply follow the lowest-cost neighboring cell until an $S$ is reached).

#### Routing order

One way to decide the order in which to route nets is to "order the nets in the ascending order of the [number] of pins within their bounding boxes[^zhou]." This seeks to avoid enclosing or intersecting other would-be routes by routing the least extensive nets first. In the example below, this approach is successful.

In practice, this may not lead to the best order. Other approaches require many passes: route order is chosen in some random fashion and nets are heuristically routed and 'ripped up' until all nets are routed, all possibilities are exhausted, or some time limit is exceeded.

### Demo

I constructed an approximate grid based on the PCB for this [fridge "door open" alarm](http://www.learningelectronics.net/circuits/fridge-door-open-alarm-circuit-project.html) project. The first figure below is the PCB from that project (I did not create it). The second figure is like a topological approximation. I have not preserved the scale. The node colours represent net membership, i.e. all of the light red nodes must be connected to each other through the routing process. The rightmost figure shows the final routed PCB.


[![Original PCB layout](/images/orig_t.jpg "Original PCB layout")](/images/orig.jpg)
[![My approximated model - unrouted](/images/unrouted_t.png "My approximated model - unrouted")](/images/unrouted.png)
[![Routed](/images/routed_t.png "Routed")](/images/routed.png)

Below is an animation of this grid being routed by the method described above:

- The routing order is determined by the number of pins w/in a net's bounding box (not shown)
- For each net (in order), until it is fully routed or unroutable:
    - Pick a source grid cell, $S$, at random (or, if partially routed, use the entire routed portion as a source)
    - Pick the closest destination, $T$. This is the single blue cell shown in the upper subplot
    - Propagate wave from $S$ to $T$ across the weighted grid (Top Subplot)
    - Backtrace a lowest-cost path (Bottom Subplot)

<a id="vis"><iframe id="ytplayer" type="text/html" width="640" height="390" src="http://www.youtube.com/embed/2kUMe5PkyCg?autoplay=0&origin=http://blog.iank.org" frameborder="0" allowfullscreen></iframe></a>

### Code

MATLAB code for this demo may be found on github: [iank/route1](https://github.com/iank/route1).

### References

[^zhou]: H. Zhou. *Northwestern University EECS357, Introduction to VLSI CAD. Lecture 6 [PDF Document]*. Retrieved from: [http://users.eecs.northwestern.edu/~haizhou/357/lec6.pdf](http://users.eecs.northwestern.edu/~haizhou/357/lec6.pdf)
