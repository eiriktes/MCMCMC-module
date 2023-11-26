mutable struct mcmcmc_chain
  param_actu::Vector{Float64}
  succes_actu::Float64
  etat::String
end

struct mcmcmc_initial
  data
  chains
  succes::Vector{Float64}
  probas::Vector{Float64}
  chain_nb::Vector{UInt8}
  etat::Vector{String}
  modeles::Matrix{Float64}
  nb_chaud::UInt8
  fixed
  loglikelyhood::Function
end

struct mcmcmc_result
  data
  succes::Vector{Float64}
  chain_nb::Vector{UInt8}
  iter_nb::Vector{UInt32}
  etat::Vector{String}
  modeles::Matrix{Float64}
  valide::Vector{Bool}
  nb_chaud::UInt8
  fixed
end

function mcmcmc_init(data, start_parameters, n_chains = 10, fixed = nothing)
  fixed = isnothing(fixed) ? nothing : Bool.(fixed)
  nchaud = n_chains > 30 ? 10 : [1,1,1,1,2,2,2,3,3,3,4,4,5,5,5,6,6,6,7,7,7,7,8,8,8,8,9,9,9,10][n_chains] 

  loglik = loglikelyhood(data)

  if !isa(start_parameters, VecOrMat)
    error("start_parameters should be a matrix or vector")
  end

  if isa(start_parameters , Vector)
    start_parameters = broadcast(+, zeros(length(start_parameters),n_chains), start_parameters)
  elseif size(start_parameters,2) < n_chains
    tofill = n_chains - size(start_parameters,2)
    start_parameters = hcat(start_parameters, start_parameters[: , [repeat(Base.oneto(size(start_parameters,2)), div(tofill, size(start_parameters,2))) ; Base.oneto(rem(tofill, size(start_parameters,2)))]])
  end
  
  chains = Vector{mcmcmc_chain}(undef, n_chains)
  
  for chain in Base.oneto(length(chains))
    chains[chain] = mcmcmc_chain(start_parameters[:,chain], loglik(start_parameters[:,chain]), "froid")
  end

  mcmcmc_initial(data, chains, getfield.(chains, :succes_actu), zeros(n_chains),  Vector(Base.oneto(n_chains)), getfield.(chains, :etat), start_parameters, nchaud, fixed, loglik)
end



function mcmcmc_iters(init, n_iter = 1000)
  
  # parametre de mcmc
  chains = init.chains
  n_chains = length(chains)
  n_estim = n_iter*n_chains
  n_param = size(init.modeles, 1)
  data = init.data

  # calc_logl = Vector{Number}(undef,n_estim)
  # calc_modif = Vector{Number}(undef,n_estim)
  # calc_proba = Vector{Number}(undef,n_estim)
  # calc_chain = Vector{Number}(undef,n_estim)
  # calc_iter = Vector{Number}(undef,n_estim)

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
  init.nb_chaud
  chains = init.chains
  width = 0.1
  modifier = modification(n_param,width, data)
  #time_t <- Sys.time()
  last_it = 0
  choice = rand(n_iter*n_chains)

  for i in Base.oneto(n_iter)

    # suivis de avancement du calcul
    if rem(i,floor(n_iter/20))==0
      println(i)
      # cat(paste("iteration : ",i,"\n",sep = ""))
      # cat("Expected end :", as.character(Sys.time() + ((Sys.time() - time_t)/floor(n_iter/10)) * (n_iter - i)), "\n")
    end
    
    # assure que le taux d'acceptation global des modeles soit de 0.44 (globalement parce qu'on veut que les chaines gaudes et froides aient des taux d'acceptation differents)
    if rem(i,100) == 0 && isfinite(maximum(getfield.(chains, :succes_actu)))
      acc_rate = sum(valide[1:(i)*n_chains])/(i*n_chains) # a verifier
      last_it = i
      width = width * exp(1 * (acc_rate - 0.44))
      modifier = modification(n_param,width,data)
    end

    # changement des chaines
    for chain in Base.oneto(length(chains))
      # calc_chain[(i-1)*n_chains + chain] = chain
      # calc_iter[(i-1)*n_chains + chain] = i

      # cree un nouveau modele et regarde sa likelyhood
      # test = @timed modifier(chains[chain].param_actu, chains[chain].etat)
      # modif = test.value
      #calc_modif[(i-1)*n_chains + chain] = test.time

      
      modif = modifier(chains[chain].param_actu, chains[chain].etat)
      new_param = isnothing(init.fixed) ? modif : ifelse.(init.fixed, chains[chain].param_actu, modif)
      
      # test = @timed init.loglikelyhood(new_param)
      # valeur = test.value
      # calc_logl[(i-1)*n_chains + chain] = test.time
      valeur = init.loglikelyhood(new_param)


      if isnan(valeur) || ismissing(valeur)
        valeur = -Inf
      end
      
      # test = @timed metropolis_log(chains[chain].succes_actu, valeur) # metropolis vus que probas de x -> x´ et x´ -> x est la même
      # calc_proba[(i-1)*n_chains + chain] = test.time
      # proba = test.value

      proba = metropolis_log(chains[chain].succes_actu, valeur) # metropolis vus que probas de x -> x´ et x´ -> x est la même
      
      if isnan(proba) || ismissing(proba)
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
    
    #les chaines avec le succes le plus eleve (proba ou likelyhood) deviennent des chaines chaudes
    # invperm(sortperm(A)) donne rank de A
    ordre_succes = invperm(sortperm( getfield.(chains, :succes_actu) ))

    for chain in Base.OneTo(n_chains)
      chains[chain].etat = ordre_succes[chain] > (n_chains - init.nb_chaud) ? "chaud" : "froid"
    end
    # a regler
    etat[i*n_chains .+ Vector(1:n_chains)] = getfield.(chains, :etat) #[all_modeles$chain_nb[-seq_along(all_modeles$etat)]]
    
  end

  return mcmcmc_result(data, succes, chain_nb, iter_nb, etat, modeles, valide, init.nb_chaud, init.fixed)
  #return (calc_logl, calc_modif, calc_proba, calc_chain, calc_iter)
end