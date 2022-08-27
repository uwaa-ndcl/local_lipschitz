hello_start_time = time()
using LinearAlgebra
using SparseArrays
using ArgParse
using Printf
using Parameters
using MosekTools
using Dates

include("../src/FastNDeepLipSdp.jl"); using .FastNDeepLipSdp

# Argument parsing
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      help = "A particular nnet that is loaded"
    "--tau"
      arg_type = Int
      default = 2
    "--Wknorm"
      arg_type = Float64
      default = 2.0
  end
  return parse_args(ARGS, argparse_settings)
end
args = parseArgs()

ffnet, weight_scales = loadNeuralNetwork(args["nnet"], args["Wknorm"])


