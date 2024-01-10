# define mcmc classes

# a reparer car marche pas avec structure actuelle de mcmcmc. specification de sd pose problème
# struct mcmc_glm_gaus
#     x::VecOrMat{Float64}
#     y::Vector{Float64}
# end

struct mcmc_glm_pois
    x::VecOrMat{Float64}
    y::Vector{Int16}
end

struct mcmc_glm_binom
    x::VecOrMat{Float64}
    y::Matrix{Int16}
end


# modifications des parametes

# function modification(n_params, mc_class::mcmc_glm_gaus)
#     (param, width) -> param + randn(n_params) * width
# end

function modification(n_params, mc_class::mcmc_glm_pois)
    (param, width) -> param + randn(n_params) * width
end
  
function modification(n_params, mc_class::mcmc_glm_binom)
    (param, width) -> param + randn(n_params) * width
end


# expected values from parameters

# marche pas car dernier paramettre pour sd de distrib pas accessible
# function param_transform(mc_class::mcmc_glm_gaus, fixed = nothing)
#     x = mc_class.x
#     n_vars = size(x, 2)
#     if isnothing(fixed)
#         (param) -> hcat(ones(size(x, 1), 1), x) * reshape(param[1:n_vars], :, 1)
#     else
#         (param) -> hcat(ones(size(x, 1), 1), x) * reshape(ifelse.(ismissing.(fixed[1:n_vars]), param[1:n_vars], fixed[1:n_vars]), :, 1)
#     end
# end

function param_transform(mc_class::mcmc_glm_pois, fixed = nothing)
    x = mc_class.x
    if isnothing(fixed)
        (param) -> exp.(hcat(ones(size(x, 1), 1), x) * reshape(param, :, 1))
    else
        (param) -> exp.(hcat(ones(size(x, 1), 1), x) * reshape(ifelse.(ismissing.(fixed), fixed, param), :, 1))
    end
end

function param_transform(mc_class::mcmc_glm_binom)
    x = mc_class.x
    if isnothing(fixed)
        (param) -> inv_logit.(hcat(ones(size(x, 1), 1), x) * reshape(param, :, 1))
    else
        (param) -> inv_logit.(hcat(ones(size(x, 1), 1), x) * reshape(ifelse.(ismissing.(fixed), fixed, param), :, 1))
    end
end

# loglikelyhoods

# function loglikelyhood(mc_class::mcmc_glm_gaus)
#     y = mc_class.y
#     func = param_fonc(mc_class)
#     n_params = size(x, 2) + 1
#     (param) -> sum(dnorm.(y, func(param), param[n_params]))
# end

function loglikelyhood(mc_class::mcmc_glm_pois)
    y = mc_class.y
    (eta) -> sum(dpois.(y, eta))
end

function loglikelyhood(mc_class::mcmc_glm_binom)
    y = mc_class.y
    (eta) -> sum(dbinom.(y[:,1], sum!(ones(Int, size(y,1), 1),y), eta))
end