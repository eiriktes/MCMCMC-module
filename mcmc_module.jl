module MCMC_ET

# export, using, import statements are usually here; we discuss these below
export mcmcmc_init, mcmcmc_iters, loglikelyhood, param_fonc, expand_grid

include("helper functions.jl")
include("proba_distrib.jl")
include("mc_distrib.jl")
include("mcmcmc.jl")

end