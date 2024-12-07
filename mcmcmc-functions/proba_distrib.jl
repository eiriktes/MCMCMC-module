

function lfactorial(x)
  if x <= 256
    return lfactorial_tab[x+1]
  else
    return Float64(log(factorial(big(x))))
    # approx
    #(x - 1/2) * log(x) - x + (1/2) * log(2 * π) + 1/(12 * x)
    # https://www.johndcook.com/blog/2010/08/16/how-to-compute-log-factorial/
    # not as precise as in article ??? to check
  end  
end


function lbinomial(size, x)
  if  x > size
    error("x cannot be greater than size")
  elseif x <= 256 && size <= 256
    return lbinomial_tab[size+1][x+1]
  else
    return Float64(log(binomial(big(size),BigInt(x))))
  end
end




function dpois(x, lambda)
  if x == 0 && lambda == 0
    return 0
  end
  x * log(lambda) - lambda - lfactorial(x)
  # ifelse.(x .== 0 .&& lambda .== 0, 0, x .* log.(lambda) .- lambda .- lfactorial.(x))
end



function dbinom(x, size, p)
  if p == 0 && x == 0
    return 0
  elseif p == 1 && x == size
    return 0
  end
  lbinomial(size, x) + x * log(p) + (size - x) * log(1-p)

  # ifelse.((p .== 0 .&& x .== 0) .|| (p .== 1 .&& x .== size), 0, lbinomial.(size, x) .+ x .* log.(p) .+ (size .- x) .* log.(1 .- p))
end

function dnorm(x, mean, sd)
  # log(1/(sd*sqrt(2*π)) * exp(-1/2 * ((x-mean)/sd)^2))
  # log(sd) + log(sqrt(2*π)) + -1/2 * (x-mean)^2/sd^2
  - log(sd) - 0.9189385332046727 - 1/2 * ((x-mean)/(sd))^2 # ~ 2* plus vite que premiere ligne
end




# metropolis <- function(px1,px2){
#   min(1,px2/px1)
# }

# glauber <- function(px1,px2){
#   px2/(px1+px2)
# }

function metropolis_log(px1,px2)
  exp(min(0,px2-px1))
end
# metropolis hastings en avec probas en log ça donne prob = exp( p(x)+P(x´ -> x) - ( p(x´)+P(x -> x´) ) )

