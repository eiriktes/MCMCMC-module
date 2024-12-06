# define mcmc classes

# a reparer car marche pas avec structure actuelle de mcmcmc. specification de sd pose problème
struct mcmc_glm_gaus
    x::VecOrMat{Float64}
    y::Vector{Float64}
end

struct mcmc_glm_pois
    x::VecOrMat{Float64}
    y::Vector{Int16}
end

struct mcmc_glm_binom
    x::VecOrMat{Float64}
    y::Matrix{Int16}
end

mcmc_glm = Union{mcmc_glm_gaus, mcmc_glm_pois,mcmc_glm_binom}

# modifications des parametes

function modification(n_params, mc_class::mcmc_glm_gaus)
    (param, width) -> param + randn(n_params) * width
end


function modification(n_params, mc_class::mcmc_glm_pois)
    (param, width) -> param + randn(n_params) * width
end
  
function modification(n_params, mc_class::mcmc_glm_binom)
    (param, width) -> param + randn(n_params) * width
end


# marche pas car dernier paramettre pour sd de distrib pas accessible
function param_transform(mc_class::mcmc_glm_gaus, fixed = nothing)
    x = mc_class.x
    n_vars = size(x, 2)+1
    if isnothing(fixed)
        (param) -> [hcat(ones(size(x, 1), 1), x) * reshape(param[1:n_vars], :, 1), exp(param[n_vars + 1])]
    else
        (param) -> [hcat(ones(size(x, 1), 1), x) * reshape(ifelse.(ismissing.(fixed[1:n_vars]), param[1:n_vars], fixed[1:n_vars]), :, 1), ifelse(ismissing(fixed[n_vars + 1]), exp(param[n_vars + 1]), fixed[n_vars + 1])]
    end
end

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

function loglikelyhood(mc_class::mcmc_glm_gaus)
    y = mc_class.y
    (transformed) -> sum(dnorm.(y, transformed[1], transformed[2]))
end

function loglikelyhood(mc_class::mcmc_glm_pois)
    y = mc_class.y
    (eta) -> sum(dpois.(y, eta))
end

function loglikelyhood(mc_class::mcmc_glm_binom)
    y = mc_class.y
    (eta) -> sum(dbinom.(y[:,1], sum!(ones(Int, size(y,1), 1),y), eta))
end

function normal_prior(mc_class::mcmc_glm, center = 0., width = 1.)
    nparam = typeof(data) == mcmc_glm_gaus ? size(mc_class.x,2) + 2 : size(mc_class.x,2) + 1

    width = 2 .* (width .^ 2) # normal distrib function uses 2*var

    if length(center) == 1
        center = fill(center,nparam)
    elseif length(center) != nparam
        error("center must be of length 1 or the same lenght as the number of covariates")
    end

    if length(width) == 1
        width = fill(width,nparam)
    elseif length(width) != nparam
        error("width must be of length 1 or the same lenght as the number of covariates")
    end
    # plein de params
    function returnf(param)
        # log of normal distribution function with mean of center and variance of width
        # log(pi) = 1.1447298858494002
        sum(.- (log.(width) .+ 1.1447298858494002) ./2 .- (param .- center).^2/(width))
    end
    returnf
end

function flat_prior(mc_class::mcmc_glm)
    normal_prior(mc_class, 0., 100.)
end
