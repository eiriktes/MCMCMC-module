function expand_grid(list)
    simple_list = unique.(list)
    exp_grid = Matrix(undef, prod(length.(simple_list)), length(simple_list))
    combination = collect(Iterators.product(tuple(simple_list...)...))
    for i in 1:length(combination)
        exp_grid[i,:] = [combination[i]...]
    end
    return exp_grid
end

function logspace_add(logx, logy)
  max(logx,logy) + log1p(exp(- abs(logx - logy) ))
end

test_list = [40,33,60]
test_val = 92
min.(max.(0, test_val .- vcat(0,cumsum(test_list)[1:3-1])), test_list)

test_mod = [50,30,10] .> test_list
test_mod == zeros(Bool,3)

function expand_grid_2(list, value)
  nval = length(list)
  if sum(list) < value
     return Matrix{Int32}(undef,0,nval)
  end
  final = min.(max.(0, value .- vcat(0,cumsum(list)[1:nval-1])), list)
  start = reverse(list)
  start = min.(max.(0, value .- vcat(0,cumsum(start)[1:nval-1])), start)
  reverse!(start)
  exp_grid = Matrix{Int32}(undef, 1, nval)
  exp_grid[1,:] = start
  curent = start
  while curent != final
    curent[[nval-1,nval]] += [1,-1]
    modif = curent .> list
    if curent[nval] < 0
      modif[nval-1] = true
      curent[nval] = 0
    end
    if modif != zeros(Int8,nval)
      curent[nval] = 0
      while modif != zeros(Int8,nval)
        curent[1:nval-2] .+= modif[2:nval-1]
        curent[modif] .= 0
        modif = curent .> list
      end
      
      curent .+= reverse(min.(max.(0, (value - sum(curent)) .- vcat(0,cumsum(reverse(list))[1:nval-1])), reverse(list)))
    end
    exp_grid = vcat(exp_grid, reshape(curent,1,nval))
  end 
  return exp_grid
end


