# MCMC engine

Welcome to my personnal attempt in building a Markov Chain Monte Carlo engine for parameter estimations. This project is mostly for fun and learning about models. The core of the engine is in the `mcmcmc_init` and `mcmcmc_iters` functions. To work they need data in a `mcmc_data` struct subtype.

I implemented estimation of glm models for models with gausian, poisson and binomial error distributions. I also implemented methods for direct estimation of matrix models in population dynamics. I am currently working on implementing estimation for capture-mark-recapture models starting with the Cormack-Jolly-Seber model.

Planned improvements :
- Allowing to set the value of some parameters before starting the estimation (usefuk for performance of complex models).
- Finish implementing the CJS model.
- Implementing a multi state capture-mark-recapture model such as Jolly-move models.
- Actually have the metropolis coupled MCMC algorithm.
