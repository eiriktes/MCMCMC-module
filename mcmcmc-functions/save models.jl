
function save_mc_model(resultat::mcmcmc_result, filename::String)
    nchains = length(unique(resultat.chain_nb))
    niter = Int((length(resultat.valide)/nchains)-1)
    text = "$nchains chains for $niter iteration
$(convert(Int16 ,resultat.nb_chaud)) hot chains
model of type $(typeof(resultat.data))
ratio valide = $(sum(resultat.valide)/length(resultat.valide))\n
fixed ::$(resultat.fixed)
succes ::$(resultat.succes)
chain_nb ::$(resultat.chain_nb)
iter_nb ::$(resultat.iter_nb)
etat ::$(resultat.etat)
modeles ::$(resultat.modeles)
valide ::$(resultat.valide)
data ::$(resultat.data)"
    open(filename, "w") do io
        write(io, text)
    end
end

function load_mc_model(filename::String)
    text = read(filename, String)
    println(split(text, "\n\n")[1])  
    model = split(split(text, "\n\n")[2], "\n")

    nb_chaud = convert(UInt8, eval(Meta.parse(split(split(text,"\n")[1], " ")[1])))

    fixed = eval(Meta.parse(split(model[1],"::")[2]))
    succes = eval(Meta.parse(split(model[2],"::")[2]))
    chain_nb = eval(Meta.parse(split(model[3],"::")[2]))
    iter_nb = eval(Meta.parse(split(model[4],"::")[2]))
    etat = eval(Meta.parse(split(model[5],"::")[2]))
    modeles = eval(Meta.parse(split(model[6],"::")[2]))
    valide = eval(Meta.parse(split(model[7],"::")[2]))
    data = eval(Meta.parse(split(model[8],"::")[2]))

    mcmcmc_result(data, succes, chain_nb, iter_nb, etat, modeles, valide, nb_chaud, fixed)
end
