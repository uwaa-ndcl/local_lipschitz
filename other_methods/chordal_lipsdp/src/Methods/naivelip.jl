# Based on: https://arxiv.org/pdf/1903.01014.pdf
using Parameters
using LinearAlgebra
using Combinatorics

# Options
@with_kw struct NaiveLipOptions <: MethodOptions
  verbose::Bool = false
end

function runQuery(inst::QueryInstance, opts::NaiveLipOptions)
  total_start_time = time()

  # So simple we can just run it
  Ws = [M[:, 1:end-1] for M in inst.ffnet.Ms]
  lipconst = prod(opnorm.(Ws))

  # Now get ready to return
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

