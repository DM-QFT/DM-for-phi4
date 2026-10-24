# Generative Modeling of Stochastic Dynamics for Long-Time Evolution

Speaking script. Quantum Frontiers Mini-Workshop, RIKEN iTHEMS, October 20, 2026.

The main part runs about 26 minutes at roughly 125 words per minute. Slide numbers follow the footer N/20. **[click]** marks the next overlay of the same slide. Notes for the backup slides are at the end.

---

## Title page
*≈ 0.5 min · ends at 0.5 min*

Thank you for the invitation.

I am Yang-yang Tan from the Institute for Physics of Intelligence at the University of Tokyo. Today I will talk about learning stochastic dynamics with generative models. The idea is simple. We learn how a system moves over a short time, and use that to predict much longer times.

Part I is work with Lingxiao Wang, Gert Aarts, Diaa Habibi and Biagio Lucini. Part II is with Lingxiao Wang and Jinyang Li.

## 1/20 Outline
*≈ 0.5 min · ends at 1 min*

I start with critical phenomena. They are a clean test, because universal numbers like critical exponents are hard to reproduce by accident.

In Part I, a diffusion model samples a lattice field theory in equilibrium, including at the critical point. In Part II, the same kind of model learns the dynamics itself. We test it on critical slowing down, on phase separation, and on a colloid experiment. I close with the QCD critical point.

*[section page, click through]*

## 2/20 Critical phenomena across physics
*≈ 1 min · ends at 2 min*

Critical points appear all over physics. Top left is ethane in a closed cell. At the critical temperature the sample turns milky. Below it is a mixture of 3-picoline and heavy water. The bright vial is the one at T_c. The correlation length has grown to the wavelength of light, so the sample scatters light strongly. This is critical opalescence.

In the middle are vesicles made from cell membranes, a two-dimensional system. Near T_c they flicker with domains, and below T_c they split into two phases. On the right is the QCD phase diagram, with a critical point at the end of the first-order line.

Microscopically these systems are very different. Near T_c they behave the same way. The correlation length diverges, fluctuations appear on every scale, and observables become power laws in T minus T_c.

## 3/20 Dynamic universality
*≈ 1.75 min · ends at 3.75 min*

Universality also holds for dynamics, and there the story is richer. Hohenberg and Halperin sorted critical dynamics into classes. The table shows four of them.

Look at the cartoons. In Model A the order parameter is not conserved, so a single site can flip, like a spin in a magnet. In Model B it is conserved. It can only move, so two neighbours swap. This is the class of alloys and diffusive mixtures. In Model G the order parameter precesses around a conserved charge Q, as in antiferromagnets and the chiral transition. In Model H a conserved order parameter is carried by the flow of the fluid. This is the class of fluids and of the QCD critical end point.

All four have the same equilibrium weight, e to the minus S. Same statics, different dynamics. The relaxation time grows like xi to the power z, and conservation laws and mode couplings decide z. For Model B, z equals four minus eta.

One remark for later. A local Monte Carlo update is itself Model A dynamics. So local Markov chains suffer critical slowing down, with tau growing like xi squared. This is one reason to look at generative samplers, which draw each configuration independently.

*[section page] Let me start with equilibrium.*

## 4/20 A diffusion model turns noise back into samples
*≈ 2.75 min · ends at 6.5 min*

A generative model learns a probability distribution p of phi from samples, and then draws new ones. A diffusion model works in two directions. Going right on the strip, we add Gaussian noise to a phi-four configuration until nothing of it is left. Going back, a Langevin equation with a learned force turns noise into a new configuration.

The forward direction is trivial. The field at diffusion time t is the data plus Gaussian noise of width Sigma of t. Its density obeys a heat equation. Nothing to learn. We choose the final width larger than the spread of the data, so at t equals one we have a broad Gaussian, whatever we started from.

**[click]**

The only thing to learn is the drift of the reverse process, the gradient of log p_t. Machine learning calls it the score. For one noisy sample the target is simple. It points from the noisy field back to the clean one, and equals minus eta over Sigma. So the network learns by predicting the noise it was given. It never sees the action. It only sees samples.

**[click]**

To generate, we run the diffusion backwards with the learned drift. There is a stochastic version on the left, and a deterministic ODE on the right with half the drift and no noise. Both give the same distribution p_t at every t.

**[click]**

This one-dimensional picture, after Song and collaborators, makes it concrete. The data have two peaks. Forward in time they spread into one broad Gaussian. Red curves are noisy paths of the SDE, white curves are paths of the ODE. Backwards, both bring the Gaussian back to the two peaks. A single noisy path is irreversible. The distribution can be reversed exactly, and the ODE follows it.

For a physicist the key point is the small-time limit. As t goes to zero, the learned drift should become the gradient of log p_data, which is minus the gradient of the action. That is the drift of Parisi-Wu stochastic quantization. So a trained diffusion model contains a learned force of the theory. Whether it is the true force, we can test. I will come back to this.

## 5/20 Which theory do we sample?
*≈ 1 min · ends at 7.5 min*

The theory is scalar phi-four on a periodic lattice, with hopping parameter kappa and quartic coupling lambda. It is symmetric under phi to minus phi. Lambda to infinity gives the Ising model, lambda equals zero a free field. So it sits in the Ising class, with the fluids, mixtures and membranes from before.

The snapshots show the three phases in 2D. Below kappa_c the field fluctuates around zero. Near kappa_c there are clusters of all sizes. Above kappa_c the field is mostly positive, the symmetry is broken. On the right, the susceptibility peak grows and sharpens with L from 16 to 128, as expected at a continuous transition. The table lists the couplings we use.

## 6/20 Do the samples match HMC in every phase?
*≈ 1 min · ends at 8.5 min*

First check, against Hybrid Monte Carlo at L equals 128 in 2D. The top row is the joint distribution of the magnetization phi-bar and the action density. HMC is filled, the diffusion model dashed. One peak in the symmetric phase. Two broad lobes at the critical point. Two separate islands in the broken phase. In all three, the dashed contours sit on the filled ones, out to the 98 percent level.

The bottom panels go site by site. They show the fourth and eighth cumulants of phi at every lattice site, on the same colour scale. HMC and the diffusion model look the same in every phase. The eighth cumulant is very sensitive to the tails, so this is a demanding test.

## 7/20 Correlation function at the critical point
*≈ 1 min · ends at 9.5 min*

Universality gives a sharper test. At the critical point the propagator is a power law, G of k goes like k to the minus two plus eta. The solid line, one over k squared, is the free field. The dashed line includes the anomalous dimension.

On the left is 2D at L equals 128. Red is the diffusion model, blue is HMC. They lie on top of each other and follow eta equals one quarter, clearly off the free line. The samples carry the fingerprint of the interacting Ising fixed point. On the right is 3D. Eta is only about 0.036, so the reference lines are close, and again the two data sets coincide over the whole momentum range. Here we train and sample at the same L.

## 8/20 Train on small lattices, sample a lattice never seen
*≈ 1.75 min · ends at 11.25 min*

Large lattices are where a sampler helps most, and where training costs most. Our network has no layer tied to the lattice size, so the same weights run on any L. We can train small and sample large.

Here we train in 3D near the critical point on lattices up to 32, and sample L equals 64, which never enters training. On the left, training on 4, 8, 16 and 32 together, the red curve tracks HMC down to the smallest momentum. With only 4 and 8, the orange curve misses the infrared rise. On the right, single sizes. 32 alone comes closest, 4 alone fails completely. Several volumes together teach the infrared.

The table answers the question from the diffusion slide. We use the learned drift as the proposal of a Metropolis-adjusted Langevin step, and accept or reject with the exact action. The acceptance rate measures how close the learned force is to the true one. Training on 4 and 8 gives 0.578. Adding 16 and 32 raises it to 0.877, close to the 0.911 of a model trained directly at 64. The network learns the force of the theory, and the force transfers to a larger volume.

*[section page] So far, statics. For long-time evolution we need time.*

## 9/20 From statics to dynamics
*≈ 2 min · ends at 13.25 min*

Statics is the Boltzmann weight. It does not care how equilibrium was reached. There is no time in it. Dynamics is an equation of motion for stochastic trajectories. I assume it does not depend on time explicitly, so the system is autonomous.

Weather, proteins, driven colloids, a heavy-ion fireball. All evolve stochastically, and for most such systems the exact stochastic equation is not available. For weather and proteins, learned time evolution already exists.

**[click]**

The central object is the transition kernel, the probability to go from phi to phi-prime in a time interval delta t. For autonomous Markovian dynamics the same kernel applies at every time. Applying it n times gives the exact evolution over n delta t. That is the Chapman-Kolmogorov relation at the bottom.

**[click]**

We learn the kernel from pairs, phi at t and phi at t plus delta t, from a simulation or an experiment. Nothing else. One kernel step covers the whole delta t, far longer than the step size of a direct integrator.

**[click]**

You can view the kernel as an effective theory in real time. It integrates out the path between grid times, and every mode that relaxes faster than delta t. Since tau of k goes like k to the minus z, a longer delta t resolves only slower, longer-wavelength modes. It is a lower-energy description, with a cutoff momentum that scales like delta t to the minus one over z. If the noise has memory, we add variables, for example the noise itself, and the joint kernel composes again.

## 10/20 Model B: conserved dynamics
*≈ 1 min · ends at 14.25 min*

Our test case is Model B in two dimensions. The field relaxes by diffusion. Its time derivative is Gamma times the Laplacian of the functional derivative of the action, plus noise. The noise carries the same Laplacian, so it only moves phi around. The action is the phi-four action of Part I, now at lambda equals one.

**[click]**

In the double well, conservation means one site gains what another loses.

**[click]**

Three consequences. The mean field M stays exactly constant. The relaxation time diverges as xi to the z, with z equals four minus eta, 3.75 in 2D. And equilibrium is still e to the minus S. Model B is also used for the QCD critical point. Our reference is a direct stochastic integrator of these equations, SOSRA.

## 11/20 Conditional diffusion as the transition kernel
*≈ 1.25 min · ends at 15.5 min*

How do we learn the kernel? With a conditional diffusion model. The top row of the schematic is a physical trajectory, where domains form and grow. The bottom row is one kernel step. The present field phi_t is the condition. Starting from Gaussian noise, the model denoises into an increment delta phi. I call this diffusion time s, since t is now physical time. Adding the increment to phi_t gives the next field. Then we repeat.

The model learns the distribution of the increment from pairs alone. No drift, no noise, no free energy. It is the same network as in Part I, with phi_t as a second input. The one thing we build in is conservation. The increment sums to zero over the lattice, so M is conserved exactly.

**[click]**

We impose this with a projection P that removes the spatial mean, applied three times in the reverse process. On the initial noise, on the learned drift, and on every noise increment. No retraining is needed.

## 12/20 Equilibrium and nonequilibrium evolution
*≈ 1 min · ends at 16.5 min*

First test, at the critical point of the 2D model on a 32 by 32 lattice, which is strongly coupled. Two initial ensembles. Equilibrium configurations from HMC, and a Gaussian quench with the same mean. Neither appears in training. Each trajectory is 200 kernel steps, up to t equals 2000.

Panel a shows snapshots. In each pair the upper row is SOSRA and the lower row the learned kernel, from the same starting field. The quenched field builds up the same large-scale structure. Panel b is the power in the softest mode. In equilibrium it stays flat. After the quench it starts low, rises, and joins the equilibrium value. Points are the kernel, lines are SOSRA, and they agree all the way. Panel c shows that both conserve the mean to single-precision rounding.

## 13/20 Mode-resolved two-time correlations
*≈ 1 min · ends at 17.5 min*

Equal-time quantities do not fully test the dynamics. We need correlations in time. Rho_n correlates Fourier mode n at time t with the same mode a time separation delta t later, normalized to one at zero separation. We average over trajectories and time origins along the equilibrium run.

The plot shows the first four diagonal modes. Conserved dynamics is slow at long wavelengths. With z near four, the softest mode lives two orders of magnitude longer than mode four. Lines are SOSRA, markers the iterated kernel at delta t equals 10 per step. They fall on the same curve for every mode. From these curves we read off each relaxation time tau_n.

## 14/20 The critical exponent emerges
*≈ 1.75 min · ends at 19.25 min*

We define tau_n by rho_n dropping to one over e. Dynamic scaling says rho_n depends only on delta t over tau_n, with tau_n going like k_n to the minus z. So against x equals delta t over tau_n, all modes should fall on one curve. They do, for the kernel and for SOSRA.

The inset gives z from the slope. At L equals 32 the kernel gives 3.792, plus or minus 0.019, and SOSRA 3.812, plus or minus 0.020. The exact value is 3.75. Both sit slightly above it, which is the effective exponent of the momentum range we resolve. What matters is that kernel and SOSRA agree. The literature values on the right scatter in the same range. The kernel never saw z. It saw pairs one delta t apart, and the exponent emerges from iterating them.

**[click]**

The common curve carries more than z. It is not a simple exponential. The grey line is e to the minus x, and the data decay more slowly. We fit an exponential times a power law. A sum rule from fluctuation-dissipation fixes the initial slope through the statics, so only two parameters are free. Free fits land on the sum rule within a few percent. This shape tests the dynamics beyond one exponent.

## 15/20 Prediction on unseen L = 64
*≈ 1 min · ends at 20.25 min*

Now size extrapolation for dynamics. These kernels are trained on L equals 8, 16 and 32, and applied at 64. No L equals 64 trajectory was ever seen. For modes two to eight, the kernel gives z equals 3.855 and direct SOSRA integration gives 3.867. They agree within errors. The legend also shows a much faster way to run the reverse process. It gives the same exponent, and it is the one timed below.

The table is the wall time to evolve 512 configurations up to t equals 100 on one GH200, at matched accuracy. The kernel takes about three seconds, in and out of equilibrium. Out of equilibrium that is five to a hundred times faster than general-purpose GPU SDE implementations, and as fast as our hand-optimized SOSRA integrator. In equilibrium our SOSRA is still faster, at 1.3 seconds.

## 16/20 Phase separation on unseen L = 128
*≈ 1.25 min · ends at 21.5 min*

Next, far from equilibrium, deep in the broken phase. We start from a disordered field on a 128 lattice. At t equals 10 there is a fine labyrinth of domains. By t equals 1000 the domains are large.

We take the sign of the field and compute its equal-time correlation C. The domain size L of t is its first zero. For a conserved order parameter, phase-ordering theory predicts that C depends only on r over L of t, and that L grows like t to the one third. This is the Lifshitz-Slyozov law, observed in polymer films and membranes, as listed on the right.

This kernel saw only synthetic domain configurations on lattices up to 64. The test lattice is twice the largest training size. In panel b the curves from t equals 100 to 1000 collapse onto one scaling function. The inset shows t to the one third growth, within 2 percent of SOSRA.

## 17/20 The colloid experiment
*≈ 1.25 min · ends at 22.75 min*

So far, simulations. Now real data. In this experiment by Dieball and collaborators, polystyrene spheres move in a thin cell. An optical potential forms a periodic washboard along x, with period 4.1 micrometres and depth about 3 k_B T. There are about 1200 tracks of 30 to 60 minutes.

Two time scales. A particle rattles in a well for about 10 seconds and escapes to the next well after about 10 minutes. In the driven experiment the cell and fluid move at constant speed, which tilts the washboard.

Same recipe as before. Position pairs 10 seconds apart, from the first 10 minutes only. The condition is the phase of the particle in the potential. No potential, no diffusion constant, no drive is given. Then we iterate from the positions at 10 minutes. 300 steps, or 50 minutes, without driving. 120 steps, or 20 minutes, with driving.

## 18/20 Transport in the colloid experiment
*≈ 1.25 min · ends at 24 min*

Top row undriven, bottom row driven. Black circles are data, blue the learned kernel, orange dashed the Langevin model of the original study. The dotted line is the end of training.

Start with panel e, the mean position under driving. The prediction stays on the data for 20 minutes past training. The mean velocity agrees within 1 percent. Langevin is 6 percent off. Panel h is the mean time to move one period. Data give 168 seconds, the kernel 166.5, Langevin 157.3.

Panel a is undriven. Even without driving the data show a weak net drift. The kernel picks it up from the first 10 minutes. The Langevin model describes equilibrium, so its current is zero and it stays flat. Later the measured drift weakens, and the kernel eventually overshoots. In the remaining panels the two models do about equally well. The grey dashed curves are the instantaneous drift. The 10-second rate is smaller, and the kernel learns it directly.

## 19/20 Outlook: stochastic hydrodynamics near the QCD critical point
*≈ 1.5 min · ends at 25.5 min*

Where do we want to go? The QCD critical point. On the static side, the question is where it is. Lattice Monte Carlo has a sign problem at finite baryon chemical potential, and people attack it with lattice methods, tensor renormalization group and functional renormalization group. Experimentally, the beam energy scan searches through net-proton cumulants. They grow fast with the correlation length, the third like xi to the 4.5 and the fourth like xi to the 7.

The fireball lives only about 10 fermi over c, so it never equilibrates near the critical point. Relaxation slows down like xi to the z, and xi freezes at 2 to 3 fermi. The relevant dynamics is Model H, with z near 3, where the baryon density couples to the momentum density.

So, the same pipeline. Take pairs of n and pi from microscopic transport or from stochastic hydrodynamics, learn the kernel, iterate. From transport, nothing has to be modelled, no noise, no kinetic coefficients, no mode couplings. From stochastic hydro, we gain long steps and lattices beyond the training size.

## 20/20 Summary
*≈ 0.75 min · ends at 26.25 min*

To summarize. A diffusion model learns the drift of stochastic quantization. It matches HMC in every phase and at criticality, and transfers to larger lattices.

A conditional diffusion model learns the transition kernel of stochastic dynamics from configuration pairs.

Long-time universality emerges from short-time pairs. The dynamic exponent and the t to the one third law come from pairs a delta t apart, much shorter than the relaxation time. The same method predicted transport in a real colloid experiment.

Next, stochastic hydrodynamics near the QCD critical point.

Thank you.

---

## Backup slides for questions

### Universality
If asked why phi-four. Eight different fluids fall on one coexistence curve once temperature and density are scaled by their critical values, with beta near one third. Fluids, mixtures and uniaxial magnets share the 3D Ising exponents in the table. Lattice phi-four is the simplest member of this class.

### Score network and reference configurations
If asked about the network. It is a U-Net. It coarse-grains down to L over 8 to reach the infrared, and skip connections keep the ultraviolet. Only two things are built in. Periodic boundaries, through circular padding, and translation symmetry, through convolutions. Not even phi to minus phi. **[click]** The network learns the action, the phi to minus phi symmetry, the critical correlations, and which phase it is in. It sees only HMC configurations, 10240 in 2D and 2560 in 3D, at kappa below, at and above kappa_c.

### QCD Sampling
If asked about gauge fields or fermions. For SU(3), the diffusion runs on the group. Noise multiplies the links, which is Brownian motion on SU(3), and at t equals one the links are Haar random. The learned drift lives in the Lie algebra, so every reverse step stays in SU(3). For fermions, integrating them out gives a determinant. At small t the drift picks up the gradient of log det M, which is nonlocal. The network learns it from HMC samples, and generation needs no inverse of M.

### Training data for the kernel
If asked what the kernels saw. The critical kernel of slides 12 and 13 is trained at L equals 32 on Gaussian random fields with random amplitude, correlation length and mean. **[click]** The coarsening kernel of slide 16 is trained on random domain configurations with random domain size, wall width, amplitude and noise. Each initial field is evolved over delta t equals 10 with SOSRA. The kernel learns the rule of motion, not trajectories. Every test in the talk is an extrapolation, to other ensembles, longer times, or larger lattices. The kernels of slides 14 and 15 are different. They are trained on equilibrium pairs at L equals 8, 16 and 32.

### The scaling function
If asked about the shape of rho. The effective decay rate, minus the log derivative of rho, runs with the time scale. For a free field each Fourier mode is an eigenmode, and the rate is constant. Here phi_n mixes many relaxation modes. The rate starts at r, about 1.8, set by the statics through fluctuation-dissipation. One r for all modes is the statement that tau_n scales like G_n over k-hat squared. At long times it approaches a gap, lambda_0 about 0.27 from the tail. The sum-rule fit gives 0.33 instead, so the crossover form is only approximate.
