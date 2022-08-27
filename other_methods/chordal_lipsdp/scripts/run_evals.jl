hello_start_time = time()
using LinearAlgebra
using SparseArrays
using ArgParse
using Printf
using Parameters
using MosekTools
using Dates

include("../src/Evals.jl"); using .Evals
@printf("import loading time: %.3f\n", time() - hello_start_time)

# Argument parsing
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      help = "A particular nnet that is loaded"
    "--nnetdir"
      help = "Directory of the nnet files"
      default = "nnets"
    "--dumpdir"
      help = "Directory of where to dump things"
      default = joinpath(homedir(), "dump")
    "--skipwarmup"
      help = "Do not do the warmups"
      action = :store_true
  end
  return parse_args(ARGS, argparse_settings)
end
args = parseArgs()

# Set up some constants; set up some directories
NNET_DIR = args["nnetdir"]; @assert isdir(NNET_DIR)
RAND_NNET_DIR = joinpath(NNET_DIR, "rand"); @assert isdir(RAND_NNET_DIR)
DUMP_DIR = args["dumpdir"]; @assert isdir(DUMP_DIR)
RAND_SAVETO_DIR = joinpath(DUMP_DIR, "rand"); isdir(RAND_SAVETO_DIR) || mkdir(RAND_SAVETO_DIR)

# Some batches of random networks
rand_nnet_filepath(w, d) = "$(RAND_NNET_DIR)/rand-I2-O2-W$(w)-D$(d).nnet"
function dwτ2norm(d, w, τ)
  if w == 10
    return (1 - (d/50)) * 2.1 + (d/50) * 1.70

  elseif w == 20
    return (1 - (d/50)) * 2.1 + (d/50) * 1.65

  elseif w == 30
    return (1 - (d/50)) * 2.1 + (d/50) * 1.62
  
  else
    return (1 - (d/50)) * 2.1 + (d/50) * 1.61
  end
end

# The versions common to both ChordalSdp and LipSDP
RAND_W10 = [(rand_nnet_filepath(10, d), [(τ, dwτ2norm(d, 10, τ)) for τ in 0:15]) for d in 5:5:50]
RAND_W20 = [(rand_nnet_filepath(20, d), [(τ, dwτ2norm(d, 20, τ)) for τ in 0:15]) for d in 5:5:50]
RAND_W30 = [(rand_nnet_filepath(30, d), [(τ, dwτ2norm(d, 30, τ)) for τ in 0:15]) for d in 5:5:50]
RAND_W40 = [(rand_nnet_filepath(40, d), [(τ, dwτ2norm(d, 30, τ)) for τ in 0:06]) for d in 5:5:50]
RAND_W50 = [(rand_nnet_filepath(50, d), [(τ, dwτ2norm(d, 30, τ)) for τ in 0:06]) for d in 5:5:50]

ALL_RAND = [RAND_W10; RAND_W20; RAND_W30; RAND_W40; RAND_W50]

SMALL_RAND = [(rand_nnet_filepath(10, d), [(τ, dwτ2norm(d, 10, τ)) for τ in 0:2]) for d in [5;10;15]]

# Only include depths <= 25
FASTLIP_RAND = [
  RAND_W10[1]; RAND_W20[1]; RAND_W30[1]; RAND_W40[1]; RAND_W50[1];
  RAND_W10[2]; RAND_W20[2]; RAND_W30[2]; RAND_W40[2]; RAND_W50[2];
  RAND_W10[3]; RAND_W20[3]; RAND_W30[3]; RAND_W40[3]; RAND_W50[3];
  RAND_W10[4]; RAND_W20[4]; RAND_W30[4]; RAND_W40[4]; RAND_W50[4];
  RAND_W10[5]; RAND_W20[5]; RAND_W30[5]; RAND_W40[5]; RAND_W50[5];
 ]

# Run a batch
function runBatch(batch, method, saveto_dir; mosek_opts = EVALS_MOSEK_OPTS)
  batch_size = length(batch)
  results = Vector{Any}()
  for (i, (nnet_filepath, τnorm_pairs)) in enumerate(batch)
    iter_start_time = time()
    println("About to run [$(i)/$(batch_size)]: $(nnet_filepath)")
    if method == :lipsdp
      runNNetLipSdp(nnet_filepath, τnorm_pairs, mosek_opts=mosek_opts, saveto_dir=saveto_dir)
    elseif method == :chordalsdp
      runNNetChordalSdp(nnet_filepath, τnorm_pairs, mosek_opts=mosek_opts, saveto_dir=saveto_dir)
    elseif method == :fastlip
      runNNetFastLip(nnet_filepath, saveto_dir=saveto_dir)
    else
      error("unrecognized method: $(method)")
    end
    @printf("----------- iter done in time: %.3f\n", time() - iter_start_time)
  end
  return results
end

# Shortcut for rand
function runRandBatch(batch, method; mosek_opts = EVALS_MOSEK_OPTS)
  return runBatch(batch, method, RAND_SAVETO_DIR, mosek_opts=mosek_opts)
end

# Do a warmup of the stuff
if !args["skipwarmup"]
  println("warming up ...")
  warmup(verbose=true)
end

@printf("repl start time: %.3f\n", time() - hello_start_time)

if !(args["nnet"] isa Nothing)
  ffnet = loadNeuralNetwork(args["nnet"])
  unscaled_ffnet = loadNeuralNetwork(args["nnet"])
  scaled_ffnet, weight_scales = loadNeuralNetwork(args["nnet"], 2.0)
end

run_rand_lipsdp() = runRandBatch(ALL_RAND, :lipsdp)
run_rand_chordal() = runRandBatch(ALL_RAND, :chordalsdp)
run_rand_fastlip() = runRandBatch(FASTLIP_RAND, :fastlip)

