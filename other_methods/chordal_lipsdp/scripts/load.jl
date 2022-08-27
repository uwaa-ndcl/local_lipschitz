start_time = time()

using LinearAlgebra
using ArgParse
using Printf

include("../src/FastNDeepLipSdp.jl"); using .FastNDeepLipSdp

#
function parseArgs()
  argparse_settings = ArgParseSettings()
  @add_arg_table argparse_settings begin
    "--nnet"
      arg_type = String
      required = true
  end
  return parse_args(ARGS, argparse_settings)
end

args = parseArgs()

@printf("import loading time: %.3f\n", time() - start_time)

nnet = Utils.NNet(args["nnet"])
ffnet = loadNeuralNetwork(args["nnet"])

scaled_ffnet, αs = loadNeuralNetwork(args["nnet"], 2.0)

Ws = [M[:, 1:end-1] for M in ffnet.Ms]
