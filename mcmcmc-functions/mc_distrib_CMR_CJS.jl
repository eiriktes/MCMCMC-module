

# dans le struct,
#   avoir un bitmatrix avec les histories
#   nocc = nombre occasions
#   vasleurs pour variables de groupement des individus
#   tableau avec index des probas et valeurs des diff variables (min time et group)
#       A modifier en deux tableaux pour facteur/num ou dataframe
#   nlevel : nombre de niveaux pour chaque variable dans covartable (pas index donc length = size(table,2)-1)
#   usebase : si on utilise dans le modele les params de base necessaire a la formation du tableau (ici juste time)
struct mcmc_CJS
    observations::BitMatrix
    nocc::UInt8
    group::VecOrMat{UInt8}
    covartablesurv::Matrix{UInt8}
    nlevelsurv::Vector{UInt8}
    usebasesurv::Bool
    covartabledetect::Matrix{UInt8}
    nleveldetect::Vector{UInt8}
    usebasedetect::Bool
end


function BaseCJS(data::BitMatrix, group::Vector{UInt8}, nchain)
    nocc = size(data,2)
    ngroup = length(unique(group))
    matrindexes = LinearIndices(Matrix{UInt8}(undef, ngroup*2, nocc))
    covarsurv = Matrix{UInt8}(undef, nocc*ngroup, 3)
    covarsurv[:,2] = repeat(1:nocc, ngroup)
    covarsurv[:,3] .= repeat(1:ngroup, inner = nocc)
    covarsurv[:,1] = mapslices((ligne) -> matrindexes[ligne[2], ligne[1]], covarsurv[:,2:3], dims = 2)
    covardetect  = Matrix{UInt8}(undef, nocc*ngroup, 3)
    covardetect[:,2] = repeat(1:nocc, ngroup)
    covardetect[:,3] .= repeat(1:ngroup, inner = nocc)
    covardetect[:,1] = mapslices((ligne) -> matrindexes[ligne[2]+ngroup, ligne[1]], covardetect[:,2:3], dims = 2)
    data_CJS = mcmc_CJS(data,
                        nocc,
                        group,
                        covarsurv,
                        [nocc, ngroup],
                        false,
                        covardetect,
                        [nocc, ngroup],
                        false)
    params = zeros((ngroup)*2)

    mcmcmc_init(data_CJS, params, nchain)
end


function PossibleHistories(history::BitVector)
    # trouve toutes les types de survies différentes qui donnent les observations dans history
    historylength = length(history)
    startObs = findfirst(history)

    # npossstart : nb of possible start positions
    # npossstart = startObs 
    # parce que si le premiere observation est en occasion 3, les possibilites pour les occasion 1 et 2 sont 00, 01 ou 11
    # replaced by startObs

    possstart = BitMatrix(undef, startObs, startObs-1)
    for i in 1:startObs
        possstart[i,:] = vcat(repeat([0],startObs-i), repeat([1],i-1))
    end


    endObs = findlast(history)
    npossend = historylength - endObs + 1 
    possend = BitMatrix(undef, npossend, historylength - endObs)
    for i in 0:(npossend-1)
        possend[i+1,:] = vcat(repeat([1],i), repeat([0],historylength - endObs-i))
    end


    nposs = startObs * npossend
    result = BitMatrix(undef, nposs, historylength)
    result[:,startObs:endObs] .= 1
    result[:,1:(startObs-1)] = repeat(possstart, npossend, 1)
    result[:,(endObs+1):historylength] = repeat(possend, inner = (startObs, 1))
    result    
end


function modification(n_params, mc_class::mcmc_CJS)
    # plein de params
  function returnf(param, width)
    # que des probas (survie et detection)

    #unif en transformé
    param .- 0.5 .* width .+ rand(n_params) .* width
  
  end
  returnf
end



# reçoit un vecteur de paramettres 
#   d'abord tous les paramettres de surv puis ceux de detect
# rend une matrice avec probas
#   colonnes = capture occasions
#   lignes, par groupe de 4
#       [1,3] + 4*i for survival and 1-survival pour groupe (i+1)
#       [2,4] + 4*i  for detection and 1-detection pour groupe (i+1)

function param_transform(mc_class::mcmc_CJS, fixed = nothing)
    nocc = mc_class.nocc
    ngroup = length(unique(mc_class.group))
    if mc_class.usebasesurv
        variablessurv = mc_class.covartablesurv
        nlevelssurv = mc_class.nlevelsurv
    else
        variablessurv = mc_class.covartablesurv[:,vcat(1, 3:size(mc_class.covartablesurv,2))]
        nlevelssurv = mc_class.nlevelsurv[2:length(mc_class.nlevelsurv)]
    end

    if mc_class.usebasedetect
        variablesdetect = mc_class.covartabledetect
        nlevelsdetect = mc_class.nleveldetect
    else
        variablesdetect = mc_class.covartabledetect[:,vcat(1, 3:size(mc_class.covartabledetect,2))]
        nlevelsdetect = mc_class.nleveldetect[2:length(mc_class.nleveldetect)]
    end
    survparindex =  cumsum(vcat(0,nlevelssurv[1:(length(nlevelssurv)-1 )]))
    detectparindex = cumsum(vcat(sum(nlevelssurv),nlevelsdetect[1:(length(nlevelsdetect)-1 )]))

    if isnothing(fixed)
        (param) -> begin
            probas = zeros(Float64, 2*ngroup, nocc)
            revprobas = zeros(Float64, 2*ngroup, nocc)

            etasurv = hcat(variablessurv[:,1],mapslices((X) -> sum(param[X .+ survparindex]), variablessurv[:,2:size(variablessurv,2)], dims = 2))
            etadetect = hcat(variablesdetect[:,1],mapslices((X) -> sum(param[X .+ detectparindex]), variablesdetect[:,2:size(variablesdetect,2)], dims = 2))
            etas = vcat(etasurv, etadetect)

            probas[convert.(Int, etas[:,1])] = log_inv_logit.(etas[:,2])
            revprobas[convert.(Int, etas[:,1])] = log_rev_inv_logit.(etas[:,2])
            
            return vcat(probas, revprobas)
        end
     else
        error("not done")
    #   (param) -> ifelse.(ismissing.(fixed[1:2]), vcat(inv_logit.(param[1:2]), log(1-inv_logit(param[2]))), vcat(fixed[1:2], 1-fixed[2]))
    end
end


function proba_history(history::BitVector, group::UInt8, logprobs)
    ngroup = size(logprobs, 1)÷4
    nocc = length(history)
    possHistories = PossibleHistories(history)
    loglikelyhood_h = -Inf64
    for histindex in axes(possHistories, 1)
        posshistory = possHistories[histindex,:]
        
        # detectlikely = sum(history .* logprobs[ngroup+group,:]) + sum((posshistory .& .!history) .* logprobs[ngroup*3+group, :])
        detectlikely = sum(logprobs[ngroup+group,history]) + sum(logprobs[ngroup*3+group, (posshistory .& .!history)])

        endposs = findlast(posshistory) # doesn't survive last occurence
        posshistory[endposs] = false
        survlikely = sum(logprobs[group, posshistory])
        if(!posshistory[nocc-1])
            survlikely = survlikely + logprobs[ngroup*2+group,endposs]
        end
        
        loglikelyhood_h = logspace_add(loglikelyhood_h, survlikely + detectlikely)

        
    end
    loglikelyhood_h
end


function loglikelyhood(mc_class::mcmc_CJS)
    ObsHistories = mc_class.observations
  
    stringhist = string.(hcat(convert(Matrix{UInt8}, ObsHistories), mc_class.group))
    bitstringhist = vec(mapslices(join, stringhist, dims = 2))
    uniquebitstHistories = unique(bitstringhist)
    countHistories = zeros(Int64, length(uniquebitstHistories))
    for i in eachindex(uniquebitstHistories)
        countHistories[i] = sum(bitstringhist .== uniquebitstHistories[i])
    end
    
    uniqueHistories = BitMatrix(undef,length(uniquebitstHistories), size(ObsHistories,2))
    uniquegroup = Vector{UInt8}(undef, length(uniquebitstHistories))
    for i in eachindex(uniquebitstHistories)
        uniquecombin = parse.(Int8,split(uniquebitstHistories[i], ""))
        uniqueHistories[i,:] = uniquecombin[1:(size(ObsHistories,2))]
        uniquegroup[i] = uniquecombin[size(ObsHistories,2)+1]
    end

    # nhistory = size(ObsHistories,1)
    nuniquehistory = size(uniqueHistories, 1)

    function loglik(probas)
        loglikelyhood = 0.
        for histoindex in 1:nuniquehistory
            loglik = proba_history(uniqueHistories[histoindex,:], uniquegroup[histoindex], probas)
            loglikelyhood += loglik * countHistories[histoindex] # les likelyhood de différents individus se multiplient donc somme des loglik (c'est un et)
        end
        loglikelyhood
    end
    loglik
end


function flat_prior(mc_class::mcmc_CJS)
    # plein de params
    function returnf(param)
        # equivalent de unif en [0:1] est inverse logit en [-inf:inf]
        # derive de inverse logit pour probas des valeurs de param
        # exp.(param) ./ (1 .+ exp.(param)).^2
        # transforme en logprob apres
        sum(param .- log1p.(exp.(param)) .* 2)
        
    end
    returnf
end


function read_mark_input(filename::AbstractString)
    text = split(read(filename, String), "\n")
    no_title = true
    let nocc, ngroup, histid, histories, group_nb
        for line in text
            if no_title
                m = match(r" proc title ; [\w ]+ occasions= (?<nocc>\d+) groups= (?<ngroup>\d+) [\w\d =]* NoHist hist= (?<nhist>\d+)", line)
                if !isnothing(m)
                    no_title = false
                    nocc = parse(Int,m[:nocc])
                    ngroup = parse(Int,m[:ngroup])
                    histid = 0
                    histories = Vector{String}(undef,parse(Int,m[:nhist]))
                    group_nb = Matrix{Int}(undef,(parse(Int,m[:nhist]), ngroup))
    
                end
            else
                
                m = match(Regex(" (\\d{$nocc})"*repeat(" (\\d+)",ngroup)), line)
                if !isnothing(m)
                    histid += 1
                    histories[histid] = m[1]
                    for i in 1:ngroup
                        group_nb[histid,i] = parse(Int,m[i+1])
                    end
                end
            end
        end
    
    
    
        if no_title
            error("no title in file")
        end

        hist = BitMatrix(undef, (sum(group_nb), nocc))
        group = Vector{Int}(undef, sum(group_nb))
        curindex = 0
        for i in eachindex(histories)
            for j in axes(group_nb,2)
                hist[curindex .+ (1:group_nb[i,j]),:] .= repeat(transpose(parse.(Int,split(histories[i], ""))), group_nb[i,j])
                group[curindex .+ (1:group_nb[i,j])] .= j
                curindex = curindex + group_nb[i,j]
            end
        end
        (hist, group)
    end  
end 