# Based on: https://arxiv.org/pdf/1903.01014.pdf
using Parameters
using LinearAlgebra
using Combinatorics

# Options
@with_kw struct CpLipOptions <: MethodOptions
  verbose::Bool = false
end

# The index partitions (m = K for our NeuralNetwork)
function Jmks(m::Int, k::Int)
  @assert 1 <= k <= m - 1
  return collect(combinations(1:m-1, k))
end

# Calculating the product
function σjmk(Ws::Vector{MatF64}, jmk::VecInt, m::Int)
  @assert length(Ws) == m
  # The lower and upper bounds of the products in (4.2)
  lbs = [1; jmk .+ 1] # [1,    j[1]+1, j[2]+1, j[3]+1, ..., j[k]+1]
  ubs = [jmk; m]      # [j[1], j[2],   j[3],   j[4], ...,   m]
  @assert all(lbs .<= ubs)
  tmp = 1
  for (lb, ub) in zip(lbs, ubs); tmp *= opnorm(prod(reverse(Ws[lb:ub]))) end
  return tmp
end

# Calculate the β stuff
function βjmk(αs::VecF64, jmk::VecInt, m::Int)
  @assert length(αs) == m
  tmp = 1.0
  for j in 1:m
    if j in jmk
      tmp *= αs[j]
    else
      tmp *= 1 - αs[j]
    end
  end
  return tmp
end

# Calculate the θm value; recall that m = K in our case
function runCpLip(ffnet::NeuralNetwork, opts::CpLipOptions)
  Ws = [M[:, 1:end-1] for M in ffnet.Ms]

  # The αs should be more smartly picked depending on ffnet.activ
  αs = [0.5 for k in 1:ffnet.K]
  m = ffnet.K

  # The β_{m, ∅} * ||WK * ... * W1|| term
  tmp = βjmk(αs, VecInt([]), m) * opnorm(prod(reverse(Ws)))
  
  # The outer sum
  for k in 1:m-1
    # The inner sum
    for jmk in Jmks(m, k)
      tmp += βjmk(αs, jmk, m) * σjmk(Ws, jmk, m)
    end
  end
  return tmp
end

# Call this
function runQuery(inst::QueryInstance, opts::CpLipOptions)
  total_start_time = time()

  # Call the stuff
  lipconst = runCpLip(inst.ffnet, opts)
  values = Dict(:lipconst => lipconst)
  total_time = time() - total_start_time
  if opts.verbose
    @printf("\ttotal time: %.3f\n", total_time)
  end

  return QuerySolution(
    objective_value = lipconst,
    values = values,
    summary = nothing,
    termination_status = "OPTIMAL",
    total_time = total_time,
    setup_time = 0.0,
    solve_time = 0.0)
end

