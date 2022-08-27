using Parameters
using LinearAlgebra
using JuMP
using MosekTools
using Dualization
using Printf

# Default options for Mosek
LIPSDP_DEFAULT_MOSEK_OPTS =
  Dict("QUIET" => true)

# Options
@with_kw struct LipSdpOptions <: MethodOptions
  τ::Int = 0; @assert τ >= 0
  include_default_mosek_opts::Bool = true
  mosek_opts::Dict{String, Any} = Dict()
  use_dual::Bool = true
  verbose::Bool = false
end

# Set up the model for the solver call
function setup!(model, inst::QueryInstance, opts::LipSdpOptions)
  setup_start_time = time()
  vars = Dict()

  # Set up the variable where γ = [γt; γlip]
  γdim = γlength(opts.τ, inst.ffnet)
  γ = @variable(model, [1:γdim])
  vars[:γ] = γ
  @constraint(model, γ[1:γdim] .>= 0)

  # Set up the LMI
  Z = makeZ(γ, opts.τ, inst.ffnet)
  @constraint(model, -Z in PSDCone())
  @objective(model, Min, γ[end]) # γ[end] is γlip

  # Return information
  setup_time = time() - setup_start_time
  return model, vars, setup_time
end

# Run the query and return the solution summary
function solve!(model, vars, opts::LipSdpOptions)
  optimize!(model)
  summary = solution_summary(model)
  values = Dict()
  for (k, v) in vars; values[k] = value.(v) end
  return summary, values
end

# The interface to call
function runQuery(inst::QueryInstance, opts::LipSdpOptions)
  total_start_time = time()

  # Set up model and add solver options, with the defaults first
  model = opts.use_dual ? Model(dual_optimizer(Mosek.Optimizer)) : Model(Mosek.Optimizer)
  pre_mosek_opts = opts.include_default_mosek_opts ? LIPSDP_DEFAULT_MOSEK_OPTS : Dict()
  todo_mosek_opts = merge(pre_mosek_opts, opts.mosek_opts)
  for (k, v) in todo_mosek_opts; set_optimizer_attribute(model, k, v) end

  # Setup and solve
  _, vars, setup_time = setup!(model, inst, opts)
  summary, values = solve!(model, vars, opts)
  total_time = time() - total_start_time
  if opts.verbose
    @printf("\tsetup time: %.3f \tsolve time: %.3f \ttotal time: %.3f \tvalue: %.3e (%s)\n",
            setup_time, summary.solve_time, total_time,
            objective_value(model), string(summary.termination_status))
  end

  return QuerySolution(
    objective_value = objective_value(model),
    values = values,
    summary = summary,
    termination_status = string(summary.termination_status),
    total_time = total_time,
    setup_time = setup_time,
    solve_time = summary.solve_time)
end

