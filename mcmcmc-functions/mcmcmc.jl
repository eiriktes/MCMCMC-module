
mutable struct mcmcmc_chain
  param_actu::Vector{Float64}
  succes_actu::Float64
  etat::String
end

struct mcmcmc_initial
  data::mcmc_data
  chains::Vector{mcmcmc_chain}
  succes::Vector{Float64}
  probas::Vector{Float64}
  chain_nb::Vector{UInt8}
  etat::Vector{String}
  modeles::Matrix{Float64}
  nb_cold::UInt8
  fixed
  loglikelyhood::Function
  transformer::Function
  prior::Function
end

struct mcmcmc_result
  data::mcmc_data
  succes::Vector{Float64}
  chain_nb::Vector{UInt8}
  iter_nb::Vector{UInt32}
  etat::Vector{String}
  modeles::Matrix{Float64}
  valide::Vector{Bool}
  nb_cold::UInt8
  fixed
end

function mcmcmc_init(data::mcmc_data, start_parameters, n_chains = 10, fixed = nothing, prior = nothing)
  ncold = n_chains > 25 ? 5 : [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4][n_chains] 

  transformer = param_transform(data, fixed)
  loglik = loglikelyhood(data)
  prior_fun = isnothing(prior) ? flat_prior(data) : prior

  if !isa(start_parameters, VecOrMat)
    error("start_parameters should be a matrix or vector")
  end

  if !isnothing(fixed)
    error("not ready")
  end

  # if !isnothing(fixed)
  #   if isa(start_parameters , Vector)
  #     start_parameters = ifelse.(ismissing.(fixed), start_parameters, fixed)
  #   else
  #     fixed_m = broadcast(+, zeros(length(start_parameters),n_chains), fixed)
  #     start_parameters = ifelse.(ismissing.(fixed), start_parameters, fixed_m)
  #   end
  # end

  if isa(start_parameters , Vector)
    start_parameters = broadcast(+, zeros(length(start_parameters),n_chains), start_parameters)
  elseif size(start_parameters,2) < n_chains
    tofill = n_chains - size(start_parameters,2)
    start_parameters = hcat(start_parameters, start_parameters[: , [repeat(Base.oneto(size(start_parameters,2)), div(tofill, size(start_parameters,2))) ; Base.oneto(rem(tofill, size(start_parameters,2)))]])
  else
    start_parameters = start_parameters[:, 1:n_chains]
  end
  
  chains = Vector{mcmcmc_chain}(undef, n_chains)
  
  for chain in Base.oneto(n_chains)
    chains[chain] = mcmcmc_chain(start_parameters[:,chain], loglik(transformer(start_parameters[:,chain])) + prior_fun(start_parameters[:,chain]), "chaud")
  end

  mcmcmc_initial(data, chains, getfield.(chains, :succes_actu), zeros(n_chains),  Vector(Base.oneto(n_chains)), getfield.(chains, :etat), start_parameters, ncold, fixed, loglik, transformer, prior_fun)
end



function mcmcmc_iters(init, n_iter = 1000)
  
  # parametre de mcmc
  chains = init.chains
  n_chains = length(chains)
  n_estim = n_iter*n_chains
  n_param = size(init.modeles, 1)
  data = init.data

  
  # vecteurs de resultats
  modeles = zeros(n_param, n_estim + n_chains)
  modeles[:, 1:n_chains] = init.modeles
  valide = falses(n_estim + n_chains)
  valide[1:n_chains] .= true
  succes = zeros(n_estim + n_chains)
  succes[1:n_chains] = init.succes
  chain_nb = zeros(n_estim + n_chains)
  chain_nb[1:n_chains] = init.chain_nb
  iter_nb = zeros(n_estim + n_chains)
  iter_nb[1:n_chains] .= 0
  etat = Vector{String}(undef, n_estim + n_chains)
  etat[1:n_chains] = init.etat

  # initialisation des mcmc
  accrateObjective = (0.44*n_chains-(n_chains-init.nb_cold)*0.33)/init.nb_cold
  chains = init.chains
  width = [repeat([0.5], n_chains), repeat([0.1], n_chains)] # width[1] for individual width for cold chains, width[2] for hot chains
  modifier = modification(n_param, data)
  #time_t <- Sys.time()
  choice = rand(n_iter*n_chains)

  print(join(vcat("$n_chains chains, $n_iter iterations\nProgression :\n<",repeat([" "], 24), "|",repeat([" "], 24), ">\n ")))
  for i in Base.oneto(n_iter)

    # suivis de avancement du calcul
    if rem(i,floor(n_iter/49))==0
      print("=")
      # cat(paste("iteration : ",i,"\n",sep = ""))
      # cat("Expected end :", as.character(Sys.time() + ((Sys.time() - time_t)/floor(n_iter/10)) * (n_iter - i)), "\n")
    end
    
    # assure que le taux d'acceptation global des modeles soit de 0.44 (globalement parce qu'on veut que les chaines gaudes et froides aient des taux d'acceptation differents)
    # assure que pour les chaines froides ce soit 0.33 et 0.66 pour les chaines chaudes
    if rem(i,200) == 0
      for chain in Base.oneto(n_chains)
        if isfinite(maximum(getfield(chains[chain], :succes_actu)))
          chain_index = (chain_nb .== chain) .& (iter_nb .< i) .& (iter_nb .>= i-200)
          acc_rate_cold = sum(valide[chain_index][etat[chain_index] .== "froid"])/sum(etat[chain_index] .== "froid")
          acc_rate_hot = sum(valide[chain_index][etat[chain_index] .== "chaud"])/sum(etat[chain_index] .== "chaud")
          width[1][chain] = width[1][chain] * exp(ifelse(isfinite(acc_rate_hot), acc_rate_hot, 0.33) - 0.33)
          width[2][chain] = width[2][chain] * exp(ifelse(isfinite(acc_rate_cold), acc_rate_cold, accrateObjective) - accrateObjective)
        end
      end
      
    end

    # changement des chaines
    for chain in Base.oneto(n_chains)
      # calc_chain[(i-1)*n_chains + chain] = chain
      # calc_iter[(i-1)*n_chains + chain] = i

      # cree un nouveau modele et regarde sa likelyhood
     
      # Avec fixed dans boucle
      # modif = modifier(chains[chain].param_actu, width[ifelse(chains[chain].etat == "chaud", 2, 1)][chain] )
      # new_param = isnothing(init.fixed) ? modif : ifelse.(ismissing.(init.fixed), modif, init.fixed)

      # Avec fixed en impliqué dans 
      new_param = modifier(chains[chain].param_actu, width[ifelse(chains[chain].etat == "chaud", 1, 2)][chain] )
      transformed = init.transformer(new_param)
      
      valeur = init.loglikelyhood(transformed) + init.prior(new_param)


      if isnan(valeur) || ismissing(valeur)
        valeur = -Inf
      end

      proba = metropolis_log(chains[chain].succes_actu, valeur) # metropolis vus que probas de x -> x´ et x´ -> x est la même
      
      # if the old likelyhood is null, always change model
      if !isfinite(chains[chain].succes_actu) || ismissing(chains[chain].succes_actu)
        proba = 1
      elseif !isfinite(valeur) || ismissing(valeur)
        proba = 0
      end

      if[false, true][convert(Int,ceil(choice[(i-1)*n_chains + chain]+proba))]  # verifier avec ceiling(rand()+proba) donne 1 ou 2 avec 2 a une proba egale a la valeur proba donnee
        chains[chain].param_actu = new_param
        chains[chain].succes_actu = valeur
        valide[i*n_chains + chain] = true
      end

      succes[i*n_chains + chain] = valeur
      modeles[:,i*n_chains + chain] = new_param
      chain_nb[i*n_chains + chain] = chain
      iter_nb[i*n_chains + chain] = i
    end
    
    #les chaines avec le succes le plus eleve (proba ou likelyhood) deviennent des chaines froides
    # invperm(sortperm(A)) donne rank de A
    ordre_succes = invperm(sortperm( getfield.(chains, :succes_actu) ))

    for chain in Base.OneTo(n_chains)
      chains[chain].etat = ordre_succes[chain] > (n_chains - init.nb_cold) ? "froid" : "chaud"
    end
    # a regler
    etat[i*n_chains .+ Vector(1:n_chains)] = getfield.(chains, :etat) #[all_modeles$chain_nb[-seq_along(all_modeles$etat)]]
    
  end
  print("\n")

  return mcmcmc_result(data, succes, chain_nb, iter_nb, etat, modeles, valide, init.nb_cold, init.fixed)
    
end