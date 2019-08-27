# Mattia's ML tools
Yet another Matlab implementation of Gaussian Mixture Models, Hidden Markov Models and correlated algorithms.
Built for learning, used for my HRI'18 paper referenced below.

The code uses two libraries from Tom Minka:
- fastfit: https://github.com/tminka/fastfit
- lightspeed: https://github.com/tminka/lightspeed

In detail:
- Gaussian Mixture Models (GMM): Probability, gradient and entropy computation
- HMM: Inference and Learning for HMM (MLE, MAP only on multivariate normal emission probabilities)
- Partially HMM (PHMM): Inference and Learning (MLE, MAP only on multivariate normal emission probabilities)
- Multivariate Normal distribution (MVN): MLE, MAP, posterior predictive, entropy computation, gradient evaluation
- Multivariate T Student distribution (MVST): used for the posterior predictive of MVNs, Laplace Approximation
- Categorical Distribution: Inference and Learning (MLE, MAP)
- Dirichlet Distribution: Inference and Learning (MLE, Weigthed MLE, Entropy, KL Divergence)
- Numerically stable: log probability space implementation

References:
- __Racca, Mattia, and Kyrki Ville. "Active Robot Learning for Temporal Task Models." Proceedings of the 2018 ACM/IEEE International Conference on Human-Robot Interaction, New York, NY, USA, 2018, pp. 123–131.__
- Minka, Tom. "Estimating a Dirichlet distribution." Technical report, MIT, 2000.
- Rabiner, Lawrence R. "A tutorial on hidden Markov models and selected applications in speech recognition." Proceedings of the IEEE 77.2 (1989): 257-286.
- Murphy, Kevin P. "Machine learning: a probabilistic perspective." MIT press, 2012
- Ramasso, Emmanuel, and Thierry Denoeux. "Making use of partial knowledge about hidden states in HMMs: an approach based on belief functions." IEEE Transactions on Fuzzy Systems 22.2 (2014) 395-405.
- Huber, Marco F., et al. "On entropy approximation for Gaussian mixture random vectors." Multisensor Fusion and Integration for Intelligent Systems, 2008. MFI 2008. IEEE International Conference on. IEEE, 2008.
