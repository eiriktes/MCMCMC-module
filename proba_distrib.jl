# function lfactorial(x)
#     if( x <= 256)
#         # make table
#     else
#         # approx
#         (x - 1/2) * log(x) - x + (1/2) * log(2 * π) + 1/(12 * x)
#         # https://www.johndcook.com/blog/2010/08/16/how-to-compute-log-factorial/
#         # not as precise as in article ??? to check
#     end
# end


function dpois(x, lambda)
    x * log(lambda) - lambda - Float64(log(factorial(big(x))))
end

function dbinom(x, size, p)
    Float64(log(binomial(big(size),big(x)))) + x * log(p) + (size - x) * log(1-p)
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
