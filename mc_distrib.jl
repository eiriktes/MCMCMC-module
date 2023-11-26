
## look random walk Gibb within metropolis

# define mcmc classes

struct mcmc_glm_pois
  x::VecOrMat{Float64}
  y::Vector{Int16}
end

struct mcmc_glm_binom
  x::VecOrMat{Float64}
  y::Matrix{Int16}
end

struct mcmc_matr_mod
  x::Matrix{Int16}
  y::Matrix{Int16}
end


# modification des paramettres

function modification(n_params, width, mc_class::mcmc_glm_pois)
  (param, etat) -> param + randn(n_params) * get(Dict("froid" => width, "chaud" => width/5), etat, width)
end

function modification(n_params, width, mc_class::mcmc_glm_binom)
  (param, etat) -> param + randn(n_params) * get(Dict("froid" => width, "chaud" => width/5), etat, width)
end

function modification(n_params, width, mc_class::mcmc_matr_mod)
  nclasses = sqrt(n_params/2)
  transition = Vector{Int16}((nclasses^2+1):n_params)

  function returnf(param, etat)
    # les param de fecondite doivent etre positifs et ceux de transition entre 0 et 1
    result = Vector{Float64}(undef, n_params)
    if etat == "froid"
      result = max.(param + randn(n_params) * 1.0*width, 0.0)
      result[transition] = width > 0.194 ? rand(length(transition)) : min.(result[transition],1.0) # if width of distribution of results is to large, use uniform probability distribution for transition probabilities
      return result
    elseif etat=="chaud"
      result = max.(param + randn(n_params) * 0.1*width, 0.0)
      result[transition] = width > 1.94 ? rand(length(transition)) : min.(result[transition],1.0)
      return result
    else 
      println("Warning : etat ni chaud ni froid")
      return rand(n_params)
    end
  end

  returnf
end


# fonctions necessaire pour loglikelyhood pour interpreter les parametres

function param_fonc(mc_class::mcmc_glm_pois)
  x = mc_class.x
  (param) -> exp.(hcat(ones(size(x, 1), 1), x) * reshape(param, :, 1))
end

function param_fonc(mc_class::mcmc_glm_binom)
  x = mc_class.x
  (param) -> inv_logit.(hcat(ones(size(x, 1), 1), x) * reshape(param, :, 1))
end

function param_fonc(mc_class::mcmc_matr_mod)
  nclasses = size(mc_class.x, 1)
  (param) -> [reshape(param[1:(nclasses^2)], nclasses, nclasses), reshape(param[(nclasses^2+1):(2*nclasses^2)], nclasses, nclasses)]
end


# loglikelyhood

function loglikelyhood(mc_class::mcmc_glm_pois)
  y = mc_class.y
  func = param_fonc(mc_class)
  (param) -> sum(dpois.(y, func(param)))
end

function loglikelyhood(mc_class::mcmc_glm_binom)
  y = mc_class.y
  func = param_fonc(mc_class)
  (param) -> sum(dbinom.(y[:,1], sum!(ones(Int, size(y,1), 1),y), func(param)))
end


function loglikelyhood(mc_class::mcmc_matr_mod)
  func = param_fonc(mc_class)
  x = mc_class.x
  y = mc_class.y

  # je sais techniquement c'est pas une fonction lambda vus qu'elle a un nom mais je sais pas faire des fonctions lambda de plusieurs lignes donc on va ignorer ça
  function loglik(param)
    nclasses = size(x,1)
    nyears = size(x,2)

    # param is a vector of parametters, first is fecundities and second is transitions. survival is assumed from transition (proba of no transition)
    # param_fonc("matrix_mod", ...) gives function that transform param into a list of two matrices, first for fecundities and second for transitions
    # for each matrix, value in [i,j] -> is fecundity/transition from class i to j

    matr_mod = func(param)
    likelyhoods = Matrix{Float64}(undef,nclasses,nyears)
    defined = BitMatrix(undef, nclasses,nyears)
    
    # fecond_max_year est le nombre max de nouveaux ind depuis chaqque categorie a chacune.
    # pour avoir nb indiv de classe j produit par la classe i l'annee t c'est fecond_max_year[(j-1)+i, t]

    # 1.2 *x +15 supperieur a qpois(0.999,x) si trop chiant, va cherches fonction de quantiles
    fecond_max_year = Matrix{Int16}(undef, nclasses^2, nyears)
    for class = Base.oneto(nclasses), year = Base.oneto(nyears)
      fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year] =  ifelse.((x[:, year] .* matr_mod[1][:,class]) .== 0, 0, ceil.(1.2 .* (x[:, year] .* matr_mod[1][:,class]) .+ 13))
    end
    trans_max_year = Matrix{Int16}(undef, nclasses^2, nyears)
    for class = Base.oneto(nclasses), year = Base.oneto(nyears)
        trans_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year] =  ifelse.((x[:, year] .* matr_mod[2][:,class]) .== 0, 0, ceil.(min.(x[:, year],1.2 .* (x[:, year] .* matr_mod[2][:,class]) .+ 11)))
    end


    for class = Base.oneto(nclasses), year = Base.oneto(nyears)
      
      # find all combination of fecundities and transitions that give size in y
      
      combi_tot = expand_grid(vcat(fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year], trans_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year]), y[class,year])

      nposs = size(combi_tot,1)

      # go through id_combi_tot and check proba of all valid combinations of transition and fecundity
      lambda = x[:,year] .* view(matr_mod[1],:,class)
      for possib in Base.oneto(nposs)

        likelyhood_possib = sum(dpois.(combi_tot[possib, 1:nclasses], lambda) + dbinom.(combi_tot[possib, nclasses+1:nclasses^2], x[:,year], matr_mod[2][:,class]))
        
        if isfinite(likelyhood_possib)
          if defined[class,year] == 0
            likelyhoods[class,year] = likelyhood_possib
            defined[class,year] = 1
          else
            likelyhoods[class,year] =  logspace_add(likelyhoods[class,year], likelyhood_possib)
          end
        end
      end

      if(defined[class,year] == 0)
        return -Inf
      end

    end
    loglikelyhood = sum(likelyhoods)
    if ismissing(loglikelyhood)
      return -Inf
    else
      return loglikelyhood
    end
  end
  loglik
end

  
