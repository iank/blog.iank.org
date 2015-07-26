Title: Reconstructing phase space and estimating maximal Lyapunov exponent from experimental time series
Date: 2015-07-26 02:11
Tags: chaos, geometry, research, experimental data
Summary: In the course of my research I needed to demonstrate that some experimental data is chaotic. This post is an example of how I reconstruct phase space from 1-D experimental data by the method of delays, plot the underlying attractor, and estimate the maximal Lyapunov exponent (characterizing the divergence of the system).

### Background

Last week I took some measurements of a system for my research and needed to show if the system was [chaotic](https://en.wikipedia.org/wiki/Chaos_theory). The measured data was a 1-dimensional time series from a [Laser Doppler Vibrometer (LDV)](http://www.polytec.com/us/products/vibration-sensors/single-point-vibrometers/complete-systems/pdv-100-portable-digital-vibrometer/). In order to show the system was chaotic I reconstructed state space using the method of delays, and estimated the maximal Lyapunov exponent of the system.

Continuous[^discrete] systems must be at least three-dimensional in order to exhibit chaos, but it's possible to reconstruct higher-dimensional state space from a 1-dimensional time series[[^packard]] by lagging the data, e.g. taking $x(t)$ and turning it into a series of vectors $\[ x(t), x(t+T), ..., x(t+nT) \]$. There are a variety of methods for determining the constant $T$, and the embedding dimension (sometimes called $m$), and there is no clear best method. Fortunately these systems are somewhat forgiving.

### Method of delays

In my experiment I generated a time series based on the [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) by integrating the Lorenz equations with one of MATLAB's Runge-Kutta ODE solvers (code [lorenz_ode.m](https://github.com/iank/lyapunov_estimation/blob/master/lorenz_ode.m) [gen_chaos.m](https://github.com/iank/lyapunov_estimation/blob/master/gen_chaos.m)).

Plotting $x$, $y$, and $z$ over $t=[0,100]$ produces a nice Lorenz attractor

![Lorenz attractor](/images/chaos/lorenz_attractor.png)

I took the $z$ variable ([lorenz_z.txt](https://raw.githubusercontent.com/iank/lyapunov_estimation/master/lorenz_z.txt)) and used the method of delays to reproduce the attractor. I just guessed at a lag value. I'll demonstrate a better method below.

![Reproduced attractor](/images/chaos/lorenz_ts_reconstruction.png)

We see that it is warped, but the underlying structure is there. For my experiment I generated a 100 Hz sinusoidal tone and added a scaled (1/50) and sped-up (100x) version of the Lorenz time series generated above. One second of data (at 204,800 samples/sec) was sent to a DAC, amplified, and used to drive a speaker coil. The vibration of the coil was measured using an LDV, producing a [time series](https://raw.githubusercontent.com/iank/lyapunov_estimation/master/lorenz_ldv.csv).

![Time series LDV data](/images/chaos/lorenz_ldv.png)

### Determining lag

Andrew Fraser and Harry Swinney[[^fraser]] give a method for determining appropriate lag for the method of delays using mutual information[^memory]. [Dr. Eric Weeks](http://www.physics.emory.edu/faculty/weeks/) wrote a [C program](http://www.physics.emory.edu/faculty/weeks//software/minfo.html) which implements their method, and you can [read more about it on his website](http://www.physics.emory.edu/faculty/weeks//research/tseries3.html).

Running this program on my LDV data produced [this signal](https://raw.githubusercontent.com/iank/lyapunov_estimation/master/lorenz_ldv.mi.csv), plotted below. From this we can see that the first minimum is around $T=770$ (discrete steps, or $770 / Fs = 3.76$ milliseconds).

![I(T) for LDV data](/images/chaos/lorenz_ldv_mi.png)

I reconstruct the attractor in MATLAB using this delay:

    lz = csvread('lorenz_ldv.csv');
    N = length(lz);
    T = 770;
    X = [lz((1:N - 2*T)), lz((1:N - 2*T) + T), lz((1:N - 2*T) + 2*T)];
    X = X(1:100000, :);  % take first ~half of data so the plot isn't too dense
    plot(X(:,1), X(:,2));
    title('Reconstructed attractor from LDV data');
    xlabel('x(t)'); ylabel('x(t+770)');

![Reconstructed attractor from LDV data](/images/chaos/lorenz_ldv_attractor.png)

This does not look like the Lorenz attractor because the system is dominated by the 100 Hz carrier. It may be possible to get a better-looking reconstruction by taking the envelope of the signal. Zooming, we see dense orbits[^control]:

![Reconstructed attractor from LDV data](/images/chaos/lorenz_ldv_attractor_zoom.png)

### Estimation of MLE

[Lyapunov exponents](https://en.wikipedia.org/wiki/Lyapunov_exponent) describe how a system expands and contracts in phase space. There is a spectrum of exponents but the maximal Lyapunov exponent (MLE, often written $\lambda_1$) characterizes the system. One of the features of chaos is exponential divergence (sensitivity to initial conditions). Two trajectories, initially arbitrarily close to each other, will diverge exponentially in phase space. The existence of a positive Lyapunov exponent is good evidence for chaos. It is also an indication of the long-term predictibility of a system: it may be specified in [nats](https://en.wikipedia.org/wiki/Nat_(unit)) per second (or bits or digits), giving the amount of time it takes for uncertainty in a system to increase by a factor of $e$ (or 2 or 10). Note that there must be *exponential* divergence for this analysis to be meaningful.

If differential equations for a system are known, it may be possible to solve for $\lambda_1$. Otherwise, it can be estimated by solving the system numerically, as in [[^benettin]]: Integrate the system for two nearby initial conditions and watch how the trajectories diverge. Renormalization is necessary as most systems are bounded (two points can only be so far away from each other in phase space).

In my case, I have a small amount of data (204800 samples), and only one trajectory in phase space. Rosenstein, et. al. [^rosenstein] present a method for estimating $\lambda_1$ in this case:

- Reconstruct the attractor using the method of delays (I use an embedding dimension of $m=10$)
- For each point $j$:
    - Find that point's nearest neighbor with the constraint that it must have at least one mean period of temporal separation. The temporal separation constraint allows us to treat our single trajectory as a collection of trajectories having separate evolutions.
    - Follow the evolution of both points, calculating the distance $d_j(i)$ between them with respect to time step $i$.
- Take the average over all points $j$: $d(i) = \mathrm{mean}(d_j(i))$
- Plot $ln(d(i))$. A straight line here indicates exponential divergence (there will likely be two regions, an initial exponential divergence and then a flat portion. This happens when phase space is bounded and there is a maximum distance, as discussed above).
- Fit a line to $ln(d(i))$.

I've [implemented this algorithm in MATLAB](https://github.com/iank/lyapunov_estimation/blob/master/rosenstein.m).

    di = rosenstein(lz, 770);
    Fs = 204800;
    figure;
    plot((1:length(di)) / Fs, log(di));
    title('Divergence for LDV data - Average distance between nearest neighbors');
    xlabel('Lag (s)');
    ylabel('ln(d_j(i))');

    %% Fit line
    x1 = 1;
    x2 = 6643;
    p = polyfit((x1:x2)'/Fs, log(di(x1:x2)),1);
    h = refline(p);
    set(h, 'LineStyle', ':');
    text(0.04, 3.4, sprintf('slope = %.2f nats/sec', p(1)));

![Divergence of LDV data](/images/chaos/lorenz_ldv_divergence.png)

### Sanity check ###

Is this a reasonable value for our system? I used this method on the original, computed[^computed] Lorenz data ($[x(t), y(t), z(t)]$) and the computed time series $[z(t), z(t+T), ..., z(t+9T)]$ .

![Divergence of original attractor](/images/chaos/lorenz_orig_divergence.png)
![Divergence of z(t) time series](/images/chaos/lorenz_z_divergence.png)

Note that the $z(t)$ data has been sped up by a factor of 100, so the estimated values for $\lambda_1$ are about equal. The [actual MLE for the Lorenz system](http://sprott.physics.wisc.edu/chaos/lorenzle.htm) is known to be about 0.9056. So there is a significant error[^error] but the values are within reason for my purpose, which is to broadly characterize a time series.

[^discrete]: It's possible to have e.g. 1-D discrete maps which are chaotic.
[^packard]: Packard, Norman H., et al. "Geometry from a time series." Physical review letters 45.9 (1980): 712.
[^fraser]: Fraser, Andrew M., and Harry L. Swinney. "Independent coordinates for strange attractors from mutual information." Physical review A 33.2 (1986): 1134.
[^memory]: One of the things that differentiates chaos from random noise is that there is long-term memory, i.e. the autocorrelation function or this mutual information metric takes a long time to decay; autocorrelation for perfect Gaussian white noise is a sharp peak at zero lag, and zero everywhere else.
[^control]: Compare to a non-chaotic signal, a perfect sinusoid + white noise, which looks "fuzzy" in phase space but the orbits always quickly return to the mean after deviation.
[^benettin]: Benettin, Giancarlo, et al. "Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory." Meccanica 15.1 (1980): 9-20.
[^rosenstein]: Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A practical method for calculating largest Lyapunov exponents from small data sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.
[^computed]: Computed from the differential equations, as opposed to *measured* data from the LDV.
[^error]: The Rosenstein, et. al. paper has several tables documenting how their method behaves for various underlying attractors, embedding dimensions, number of data points, lag value, and so on.
