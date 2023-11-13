
## look random walk Gibb within metropolis

# fonctions utiles
# logspace_add <- function(logx, logy){
#   pmax(logx,logy) + log1p(exp(- abs(logx - logy) ))
# }

# glm ---------------------------------------------------------------------

function modification(n_params, width, mc_class)
  if(mc_class == "glm_pois")
    return (param, etat) -> param + randn(n_params) * get(Dict("froid" => width, "chaud" => width/5), etat, width)
  elseif mc_class == "glm_binom"
    return (param, etat) -> param + randn(n_params) * get(Dict("froid" => width, "chaud" => width/5), etat, width)
  elseif mc_class == "matr_mod"
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
    return returnf
  else
    throw(ArgumentError("type of distribution not possible"))
  end
end

function param_fonc(mc_class, x)
  println("test")
  if mc_class == "glm_pois"
    return (param) -> exp.(hcat(ones(size(x,1),1) , x ) * reshape(param, : , 1))
  elseif mc_class == "glm_binom"
    inv_logit(eta) = exp(eta)/(1+exp(eta))
    return (param) -> inv_logit.(hcat(ones(size(x,1),1) , x ) * reshape(param, : , 1))
  elseif mc_class == "matr_mod"
    nclasses = size(x,1)
    return (param) -> [reshape(param[1:(nclasses^2)],nclasses, nclasses ), reshape(param[(nclasses^2+1):(2*nclasses^2)],nclasses, nclasses )]
  end
end


function loglikelyhood(mc_class, y, x, func)
  if mc_class == "glm_pois"
    return (param) -> sum(dpois.(y, func(param)))
  elseif mc_class == "glm_binom"
    return (param) -> sum(dbinom.(y[:,1], sum!(ones(Int, size(y,1), 1),y), func(param)))
  elseif mc_class == "matr_mod"
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
      #pois_limit = 0.999
      #fecond_max_year <- qpois(pois_limit,apply(x,2,function(X){   vapply(seq_along(X), function(I){ X[I] * matr_mod[[1]][I,]} , numeric(nclasses)) }))
      # 1.2 *x +15 supperieur a qpois(0.999,x) si trop chiant, va cherches fonction de quantiles
      fecond_max_year = Matrix{UInt16}(undef, nclasses^2, nyears)
      for class = Base.oneto(nclasses), year = Base.oneto(nyears)
        fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year] =  ceil.(1.2 .* (x[:, year] .* matr_mod[1][:,class]) .+ 15)
      end

      combi_util = Matrix{Float64}(undef,0, nclasses)
      sum_util = Vector{Float64}(undef, 0)

      for class = Base.oneto(nclasses), year = Base.oneto(nyears)
        #println(class, " : ",year)
        if(class == 1 && year == 1)
          combi_poss_fec = expand_grid(range.(0, fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year]))
          combi_util = vcat(combi_util, combi_poss_fec) 
          sum_util = vcat(sum_util,vec(sum(combi_util, dims = 2))) # collapse columns together with sum and vec to get vector
    
          
          
          # extract only combinations where the total of newborns is less than the observed one in y
          # additionnaly also removes combinations where one of the parent class gives offsprings while the parametter for fecundity of this parent is 0
         
          id_combi_fec = (sum_util .<= y[class,year]) .& (vec(sum(  (combi_poss_fec .> 0) .&   reshape(matr_mod[1][:,class] .== 0,1,nclasses) , dims = 2)) .== 0)
          combi_poss_fec = combi_poss_fec[id_combi_fec,:]
          sum_poss_fec = sum_util[id_combi_fec]
          id_combi_fec = Base.oneto(length(sum_poss_fec))
        else
          # find every combination in combi util where no values is greater than max fecond values
          id_combi_fec = Base.oneto(size(combi_util,1))[vec(sum(combi_util .> reshape(fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year],1,nclasses), dims = 2)) .== 0 ]
          nb_accounted = length(id_combi_fec)


          # si le nombre total combinaisons trouvées dans combi_util est inferieur a l'attendu, ajoute ce qu'il faut
          if prod(fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year] .+ 1) > nb_accounted
            old = size(combi_util,1)
            max_int = maximum(combi_util[id_combi_fec,:], dims = 1) # vector of max value in combiposs per class (collapses rows in one using max value)
            for combi_row in Base.oneto(nclasses)[vec(max_int) .< fecond_max_year[(class-1)*nclasses .+ Base.oneto(nclasses), year]] # add combinations not taken in acount before
              sub_combi = Vector(undef, nclasses)
              for i in Base.oneto(nclasses)[Base.oneto(nclasses) .!= combi_row]
                sub_combi[i] = unique(view(combi_util,id_combi_fec,i))
              end
              sub_combi[combi_row] = range(start = max_int[combi_row] + 1 , stop = fecond_max_year[(class-1)*nclasses .+ combi_row, year])
              new_combi = expand_grid(sub_combi)
              sum_newcomb = vec(sum(new_combi, dims = 2))
              
              combi_util = vcat(combi_util, new_combi)
              sum_util = vcat(sum_util,sum_newcomb)

              id_combi_fec = vcat(id_combi_fec, range(old, size(combi_util,1)))
              
            end
          end
          combi_poss_fec  = combi_util[id_combi_fec,:]
          sum_poss_fec = sum_util[id_combi_fec]
          
          id_combi_fec = (sum_poss_fec .<= y[class,year]) .& (vec(sum(  (combi_poss_fec .> 0) .&   reshape(matr_mod[1][:,class] .== 0, 1, nclasses) , dims = 2)) .== 0)
          combi_poss_fec = combi_poss_fec[id_combi_fec,:]
          sum_poss_fec = sum_poss_fec[id_combi_fec]
          id_combi_fec = Base.oneto(length(sum_poss_fec))
        end

        
        id_combi_tran = Base.oneto(size(combi_util,1))[vec(sum(combi_util .> reshape(x[:, year],1,nclasses), dims = 2)) .== 0]  # find every combination in combi util where no values is grater than ones in x (becaus cannot have more survivors than there were individuals)
        nb_accounted = length(id_combi_tran)

        
        if prod(x[:,year] .+ 1) > nb_accounted
          old = size(combi_util,1)
          max_int = maximum(combi_util[id_combi_tran,:], dims = 1) # vector of max value in combiposs per class
          for combi_row in Base.oneto(nclasses)[max_int .< x[:,year]] # add combinations not taken in acount before
            sub_combi = Vector(undef, nclasses)
            for i in Base.oneto(nclasses)[Base.oneto(nclasses) .!= combi_row]
              sub_combi[i] = unique(combi_util[id_combi_tran,i])
            end
            sub_combi[combi_row] = range(start = max_int[combi_row] + 1 , stop = x[combi_row,year])
            new_combi = expand_grid(sub_combi) 
            sum_newcomb = vec(sum(new_combi, dims = 2))
            
            combi_util = vcat(combi_util, new_combi)
            sum_util = vcat(sum_util,sum_newcomb)
            id_combi_tran = vcat(id_combi_tran, range(old, size(combi_util,1)))
          end
        end

        combi_poss_tran = combi_util[id_combi_tran,:]
        sum_poss_tran = sum_util[id_combi_tran]
        
        id_combi_tran = (sum_poss_tran .<= y[class,year]) .& (vec(sum(  (combi_poss_tran .> 0) .&   reshape(matr_mod[2][:,class] .== 0,1,nclasses) , dims = 2)) .== 0)
        combi_poss_tran = combi_poss_tran[id_combi_tran,:]
        sum_poss_tran = sum_poss_tran[id_combi_tran]
        id_combi_tran = Base.oneto(length(sum_poss_tran))
        

        # find all combination of fecundities and transitions that give size in y
      
        id_combi_tot = expand_grid([id_combi_fec,id_combi_tran])
        id_combi_tot = id_combi_tot[(sum_poss_fec[id_combi_tot[:,1]] + sum_poss_tran[id_combi_tot[:,2]]) .== y[class,year],:]

        
        # go through id_combi_tot and check proba of all valid combinations of transition and fecundity
        nposs = size(id_combi_tot,1)
        #println(length(id_combi_fec), " ", length(id_combi_tran))
        #println(nposs)
        for possib in Base.oneto(nposs)
          
          lambda = view(x,:,year) .* view(matr_mod[1],:,class)
          likelyhood_possib = sum(dpois.(view(combi_poss_fec,id_combi_tot[possib,1],:), lambda) + dbinom.(view(combi_poss_tran,id_combi_tot[possib,2],:), x[:,year], matr_mod[2][:,class]))
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
    return loglik
  end
end


