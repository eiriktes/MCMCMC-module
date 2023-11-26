module MCMC_ET

# export, using, import statements are usually here; we discuss these below
export mcmc_glm_pois, mcmc_glm_binom, mcmc_matr_mod, mcmcmc_init, mcmcmc_iters, loglikelyhood, param_fonc, thinning, save_mc_model, load_mc_model

include("lookuptables.jl")
include("proba_distrib.jl")
include("mc_distrib.jl")
include("mcmcmc.jl")
include("helper functions.jl")
include("save models.jl")

end