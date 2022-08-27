module FastNDeepLipSdp

using LinearAlgebra
using Printf
using Random

include("Methods/Methods.jl");
include("Utils/Utils.jl");

import Reexport
Reexport.@reexport using .Methods
Reexport.@reexport using .Utils

# The default Mosek options to use
DEFAULT_MOSEK_OPTS =
  Dict("QUIET" => true,
       "MSK_DPAR_OPTIMIZER_MAX_TIME" => 60.0 * 60 * 1, # seconds
       "INTPNT_CO_TOL_REL_GAP" => 1e-9,
       "INTPNT_CO_TOL_PFEAS" => 1e-9,
       "INTPNT_CO_TOL_DFEAS" => 1e-9)

# Solve a problem instance depending on what kind of options we give it
function solveLipschitz(ffnet::NeuralNetwork, opts::MethodOptions)
  inst = QueryInstance(ffnet=ffnet)
  soln = runQuery(inst, opts) # Multiple dispatch based on opts type
  return soln
end

# Solve a problem instance also
function solveLipschitz(ffnet::NeuralNetwork, weight_scales :: VecF64, method;
                        tau = 2,
                        mosek_opts = DEFAULT_MOSEK_OPTS,
                        verbose = true)
  # Construct the options given the method
  if method == :lipsdp
    opts = LipSdpOptions(τ=tau, mosek_opts=mosek_opts, verbose=verbose)
  elseif method == :chordalsdp
    opts = ChordalSdpOptions(τ=tau, mosek_opts=mosek_opts, verbose=verbose)
  elseif method == :naivelip
    opts = NaiveLipOptions(verbose=verbose)
  elseif method == :cplip
    opts = CpLipOptions(verbose=verbose)
  else
    error("Unrecognized method: $(method)")
  end

  # Call the above
  soln = solveLipschitz(ffnet, opts)

  # Do the scaling to get the lipconst
  if method == :lipsdp || method == :chordalsdp
    lipconst = sqrt(soln.values[:γ][end]) / prod(weight_scales)
  else
    lipconst = soln.objective_value
  end
  return soln, lipconst
end


export DEFAULT_MOSEK_OPTS
export solveLipschitz

end

