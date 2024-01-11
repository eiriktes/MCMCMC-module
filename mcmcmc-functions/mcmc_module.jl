module MCMC_ET

# export, using, import statements are usually here; we discuss these below
export mcmcmc_init, mcmcmc_iters, loglikelyhood, mcmc_glm_gaus, mcmc_glm_pois, mcmc_glm_binom, mcmc_matr_mod, save_mc_model, load_mc_model, thinning

include("lookuptables.jl")
include("proba_distrib.jl")
include("mcmcmc.jl")
include("mc_distrib_glm.jl")
include("mc_distrib_matr_mod.jl")
include("helper functions.jl")
include("save models.jl")

end