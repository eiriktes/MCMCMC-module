function logspace_add(logx, logy)
  max(logx,logy) + log1p(exp(- abs(logx - logy) ))
end


inv_logit(eta) = exp(eta) / (1 + exp(eta))


function expand_grid(list,value)
  l_list = length(list)
  position = sortperm(list)
  sort!(list)
  w = reverse(cumsum(reverse(list)))
  res = Vector{Int16}(undef,0)
  expand_grid_rec(zeros(Int16,l_list),list,w,1,value,res)
  res = reshape(res,l_list,:)
  transpose(res[invperm(position),:])
end

function expand_grid_rec(v, m, w, i, N, solutions)
  if i == length(v)  # On a rempli tous le vecteur
    v[i] = N
    append!(solutions,copy(v))
    # if N > m[i]
    #   println("0")
    # end

  elseif N == 0  # le vecteur vaut déjà N
    # met toutes les valeurs restantes à 0
    for j in range(i,length(v))
      v[j] = 0
    end
    append!(solutions,copy(v))
  else
    for x in range(max(0,N-w[i+1]),min(N, m[i]))
      v[i] = x
      expand_grid_rec(v, m, w, i+1, N-x, solutions)
    end
  end
end

function thinning(res::mcmcmc_result, out_length, start = 0)
  #n_chains = length(unique(res.chain_nb))
  n_iters = maximum(res.iter_nb)
  step = floor(Int,(n_iters+1-start)/out_length)
  thinned_i = start:step:n_iters
  thinning = zeros(Bool, length(res.iter_nb))
  for i in 1:length(res.iter_nb)
    thinning[i] = any(res.iter_nb[i] .== thinned_i)
  end

  mcmcmc_result(res.data, res.succes[thinning], res.chain_nb[thinning], res.iter_nb[thinning], res.etat[thinning], res.modeles[:,thinning], res.valide[thinning], res.nb_chaud, res.fixed)
  
end
