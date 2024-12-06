
struct mcmc_matr_mod
    x::Matrix{Int16}
    y::Matrix{Int16}
end


function modification(n_params, mc_class::mcmc_matr_mod)
  nclasses = sqrt(n_params/2)
  transition = Vector{Int16}((nclasses^2+1):n_params)

  function returnf(param, width)
    # les param de fecondite doivent etre positifs et ceux de transition entre 0 et 1
    

    # normal distrib : DONNE UN FORT BIAS SUR 0 POUR PARAM PROCHES DE 0
    # result = Vector{Float64}(undef, n_params)
    # result = max.(param + randn(n_params) * width, 0.0)
    # result[transition] = width > 0.194 ? rand(length(transition)) : min.(result[transition],1.0) # if width of distribution of results is to large, use uniform probability distribution for transition probabilities
    # return result

    # unif distrib : POUR PARAMS PROCHES DE 0 DONNE UN BIAIS qui s'éloigne de 0
    # mins = max.(0, param .- (width * 0.5))
    # maxs = param .+ (width * 0.5)
    # maxs[transition] = min.(1, maxs[transition])
    # mins .+ rand(n_params) .* (maxs .- mins)

    #unif en transformé
    param .- 0.5 .* width .+ rand(n_params) .* width
  
  end

  returnf
end

function param_transform(mc_class::mcmc_matr_mod, fixed = nothing)
    nclasses = size(mc_class.x, 1)
    # avec param as lambdas et probas
    #(param) -> [reshape(param[1:(nclasses^2)], nclasses, nclasses), reshape(param[(nclasses^2+1):(2*nclasses^2)], nclasses, nclasses)]
    #avec params as eta (ici transphorme en probas et lambda)

    if isnothing(fixed)
      (param) -> [reshape(exp.(param[1:(nclasses^2)]), nclasses, nclasses), reshape(inv_logit.(param[(nclasses^2+1):(2*nclasses^2)]), nclasses, nclasses)]
    else
      fixed_fec = fixed[1:(nclasses^2)]
      fixed_tran = fixed[(nclasses^2+1):(2*nclasses^2)]
      (param) -> [reshape(ifelse.(ismissing.(fixed_fec), exp.(param[1:(nclasses^2)]), fixed_fec), nclasses, nclasses), reshape(ifelse.(ismissing.(fixed_tran), inv_logit.(param[(nclasses^2+1):(2*nclasses^2)]), fixed_tran), nclasses, nclasses)]
    end
    #(param) -> reshape(param, nclasses, nclasses, 2)
end




# function loglikelyhood(mc_class::mcmc_matr_mod)
#   func = param_fonc(mc_class)
#   x = mc_class.x
#   y = mc_class.y

#   function loglik(param)
#     nclasses = size(x, 1)
#     nyears = size(x, 2)

#     expandt = zero(Float64)
#     probt = zero(Float64)

#     # param is a vector of parametters, first is fecundities and second is transitions. survival is assumed from transition (proba of no transition)
#     # param_fonc("matrix_mod", ...) gives function that transform param into a list of two matrices, first for fecundities and second for transitions
#     # for each matrix, value in [i,j] -> is fecundity/transition from class i to j

#     matr_mod = func(param)
#     likelyhoods = Matrix{Float64}(undef, nclasses, nyears)
#     defined = BitMatrix(undef, nclasses, nyears)

#     # fecond_max_year est le nombre max de nouveaux ind depuis chaqque categorie a chacune.
#     # pour avoir nb indiv de classe j produit par la classe i l'annee t c'est fecond_max_year[(j-1)+i, t]

#     # 1.2 *x +15 supperieur a qpois(0.999,x) si trop chiant, va cherches fonction de quantiles
#     fecond_max_year = Matrix{Int16}(undef, nclasses^2, nyears)
#     for class = Base.oneto(nclasses), year = Base.oneto(nyears)
#       fecond_max_year[(class-1)*nclasses.+Base.oneto(nclasses), year] = ifelse.((x[:, year] .* matr_mod[1][:, class]) .== 0, 0, ceil.(1.2 .* (x[:, year] .* matr_mod[1][:, class]) .+ 13))
#     end
#     trans_max_year = Matrix{Int16}(undef, nclasses^2, nyears)
#     for class = Base.oneto(nclasses), year = Base.oneto(nyears)
#       trans_max_year[(class-1)*nclasses.+Base.oneto(nclasses), year] = ifelse.((x[:, year] .* matr_mod[2][:, class]) .== 0, 0, ceil.(min.(x[:, year], 1.2 .* (x[:, year] .* matr_mod[2][:, class]) .+ 11)))
#     end


#     for class = Base.oneto(nclasses), year = Base.oneto(nyears)

#       # find all combination of fecundities and transitions that give size in y

#       combi_tot = expand_grid(vcat(fecond_max_year[(class-1)*nclasses.+Base.oneto(nclasses), year], trans_max_year[(class-1)*nclasses.+Base.oneto(nclasses), year]), y[class, year])

#       # combi_tot = expand_grid_2_3(vcat(fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year], x[:, year]), y[class,year])

#       nposs = size(combi_tot, 1)

#       # go through id_combi_tot and check proba of all valid combinations of transition and fecundity
#       lambda = x[:, year] .* view(matr_mod[1], :, class)
#       for possib in Base.oneto(nposs)

#         likelyhood_possib = sum(dpois.(combi_tot[possib, 1:nclasses], lambda) + dbinom.(combi_tot[possib, nclasses+1:nclasses^2], x[:, year], matr_mod[2][:, class]))



#         if isfinite(likelyhood_possib)
#           if defined[class, year] == 0
#             likelyhoods[class, year] = likelyhood_possib
#             defined[class, year] = 1
#           else
#             likelyhoods[class, year] = logspace_add(likelyhoods[class, year], likelyhood_possib)
#           end
#         end
#       end

#       if (defined[class, year] == 0)
#         return -Inf
#       end

#     end
#     loglikelyhood = sum(likelyhoods)
#     if ismissing(loglikelyhood)
#       return -Inf
#     else
#       return loglikelyhood
#     end
#   end
#   loglik
# end




function loglikelyhood(mc_class::mcmc_matr_mod)
  x = mc_class.x
  y = mc_class.y

  function loglik(param)
    nclasses = size(x, 1)
    nyears = size(x, 2)

    # expandt = zero(Float64)
    # probt = zero(Float64)
    # sumt = zero(Float64)


    # param is a vector of parametters, first is fecundities and second is transitions. survival is assumed from transition (proba of no transition)
    # param_fonc("matrix_mod", ...) gives function that transform param into an list of two matrices, first for fecundities and second for transitions
    # for each matrix, value in [i,j] -> is fecundity/transition from class j to i

    matr_mod = param
    likelyhoods = Matrix{Float64}(undef, nclasses, nyears)

    # fecond_max_year est le nombre max de nouveaux ind depuis chaqque categorie a chacune.
    # pour avoir nb indiv de classe j produit par la classe i l'annee t c'est fecond_max_year[(j-1)+i, t]
    
    # 1.2 *x +15 supperieur a qpois(0.999,x) si trop chiant, va cherches fonction de quantiles
    fecond_max_year = zeros(Int16, nclasses^2, nyears)# Matrix{Int16}(undef, nclasses^2, nyears)
    # for class = Base.oneto(nclasses), year = Base.oneto(nyears)
    for classfrom = Base.oneto(nclasses), classto = Base.oneto(nclasses)
      fecond_max_year[LinearIndices(matr_mod[1])[classto,classfrom], :] = ifelse.((x[classfrom, :] .* matr_mod[1][classto, classfrom]) .== 0, 0, ceil.(1.2 .* (x[classfrom, :] .* matr_mod[1][classto, classfrom] .+ 13)))
      # fecond_max_year[(class).+0:(nclasses-1).*nclasses, year] = ifelse.((x[:, year] .* reshape(matr_mod[1][class, :], :, 1)) .== 0, 0, ceil.(1.2 .* (x[:, year] .* reshape(matr_mod[1][class, :], :, 1) .+ 13)))
      
    end

    trans_max_year = zeros(Int16, nclasses^2, nyears)#Matrix{Int16}(undef, nclasses^2, nyears)
    #for class = Base.oneto(nclasses), year = Base.oneto(nyears)
    for classfrom = Base.oneto(nclasses), classto = Base.oneto(nclasses)
      trans_max_year[LinearIndices(matr_mod[2])[classto,classfrom], :] = ifelse.((x[classfrom, :] .* matr_mod[2][classto, classfrom]) .== 0, 0, min.(x[classfrom, :], ceil.(1.2 .* (x[classfrom, :] .* matr_mod[2][classto, classfrom] .+ 11))) )
      #trans_max_year[(class-1)*nclasses.+Base.oneto(nclasses), year] = ifelse.((x[:, year] .* matr_mod[2][:, class]) .== 0, 0, ceil.(min.(x[:, year], 1.2 .* (x[:, year] .* matr_mod[2][:, class]) .+ 11)))
    end


    for class = Base.oneto(nclasses), year = Base.oneto(nyears)

      # find all combination of fecundities and transitions that give size in y
      matrindx = vec(LinearIndices(matr_mod[1])[class,:])

      # test_exp = @timed expand_grid(vcat(fecond_max_year[matrindx, year], trans_max_year[matrindx, year]), y[class, year])
      # combi_tot = test_exp.value
      # expandt += test_exp.time
      combi_tot = expand_grid(vcat(fecond_max_year[matrindx, year], trans_max_year[matrindx, year]), y[class, year])


      nposs = size(combi_tot, 1)
      if nposs == 0
        return -Inf
      end

      # go through id_combi_tot and check proba of all valid combinations of transition and fecundity
      lambda = reshape(x[:, year] .* matr_mod[1][class, :], 1,nclasses) # matr_mod[1][class, :] donne un vecteur et pas une ligne de matrice


      # test_prob = @timed sum(hcat(dpois.(combi_tot[:, 1:nclasses], reshape(lambda, 1, :)), dbinom.(combi_tot[:, nclasses+1:nclasses^2], reshape(x[:, year], 1, :), reshape(matr_mod[2][:, class], 1, :))), dims = 2)
      # likelyhood_possib = test_prob.value
      # probt += test_prob.time
      likelyhood_possib = sum(hcat(dpois.(combi_tot[:, 1:nclasses], lambda), dbinom.(combi_tot[:, nclasses+1:(2*nclasses)], reshape(x[:, year], 1, nclasses), reshape(matr_mod[2][class, :], 1, nclasses))), dims = 2)
      # sort!(likelyhood_possib, rev = true)

      # test_logsum = @timed likelyhood_possib[1]
      # likelyhoods[class, year] = test_logsum.value
      # sumt += test_logsum.time
      likelyhoods[class, year] = likelyhood_possib[1]

      if nposs > 1
        for i in 2:nposs
          # likelyhoods[class, year] = logspace_add(likelyhoods[class, year], likelyhood_possib[i])
          # test_logsum = @timed logspace_add!(likelyhoods, likelyhood_possib[i], CartesianIndex(class, year))
          # sumt += test_logsum.time
          logspace_add!(likelyhoods, likelyhood_possib[i], CartesianIndex(class, year))

          # maxv + log1p(exp(minv - maxv ))
          # avec sort!(likelyhood_possib, rev = true), likelyhoods[class, year] est toujours plus grand que likelyhood_possib[i]
          # likelyhoods[class, year] += log1p(exp(likelyhood_possib[i] - likelyhoods[class, year] ))
          
        end
      end

      if !isfinite(likelyhoods[class, year])
        return -Inf
        # return (-Inf, expandt, probt, sumt)
      end

    end
    loglikelyhood = sum(likelyhoods)
    if ismissing(loglikelyhood)
      return -Inf
      # return (-Inf, expandt, probt, sumt)
    else
      return loglikelyhood
      # return (loglikelyhood, expandt, probt, sumt)
    end
  end
  loglik
end



function flat_prior(mc_class::mcmc_matr_mod)
  function returnf(param)
    # normal distrib for each parameters with variance of 2
    sum(.- (log.(4.) .+ 1.1447298858494002) ./2 .- (param).^2/(4.))
  end
  returnf
end

