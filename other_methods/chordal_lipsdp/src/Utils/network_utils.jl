
# Actually realize the activation
function makeϕ(activ::Activation)
  if activ isa ReluActivation
    return x -> max.(x, 0)
  elseif activ isa TanhActivation
    return x -> tanh.(x, 0)
  else
    error("unsupported activation $(activ)")
  end
end

# Run a feedforward net on an initial input and give the output
function runNetwork(ffnet::NeuralNetwork, x1)
  ϕ = makeϕ(ffnet.activ)
  xk = x1
  # Run through each layer
  for Mk in ffnet.Ms[1:end-1]; xk = ϕ(Mk * [xk; 1]) end
  # Then the final layer does not have an activation
  xk = ffnet.Ms[end] * [xk; 1]
  return xk
end

# Randomized sampling of ||f(x) - f(y)|| / ||x - y||
function randomizedLipschitz(ffnet::NeuralNetwork, N::Int = 500000)
  xdim1 = ffnet.xdims[1]
  xys = [begin x = 100 * randn(xdim1); y = x + randn(xdim1); (x, y) end for k in 1:N]
  diffs = [norm(runNetwork(ffnet, x) - runNetwork(ffnet, y)) / norm(x - y) for (x, y) in xys]
  return maximum(diffs)
end

# Convert NNet to NeuralNetwork
function loadNeuralNetwork(nnet_filepath::String)
  nnet = NNetParser.NNet(nnet_filepath)
  Ms = [[nnet.weights[k] nnet.biases[k]] for k in 1:nnet.numLayers]
  xdims = Vector{Int}(nnet.layerSizes)
  ffnet = NeuralNetwork(activ=ReluActivation(), xdims=xdims, Ms=Ms)
  return ffnet
end

# Scaled version of load where each Wk has a corresponding opnorm
function loadNeuralNetwork(nnet_filepath::String, Wk_opnorm::Float64)
  ffnet = loadNeuralNetwork(nnet_filepath)
  Ws, bs = [M[:,1:end-1] for M in ffnet.Ms], [M[:,end] for M in ffnet.Ms]
  αs = [Wk_opnorm / opnorm(W) for W in Ws]
  scaled_Ws = [αs[k] * Ws[k] for k in 1:ffnet.K]
  scaled_bs = [prod(αs[1:k]) * bs[k] for k in 1:ffnet.K]
  scaled_Ms = [[scaled_Ws[k] scaled_bs[k]] for k in 1:ffnet.K]
  scaled_ffnet = NeuralNetwork(activ=ffnet.activ, xdims=ffnet.xdims, Ms=scaled_Ms)
  return scaled_ffnet, αs
end

# Generate a random network given the desired dimensions at each layer
function randomNetwork(xdims::VecInt; activ::Activation = ReluActivation(), σ::Float64 = 1.0)
  @assert length(xdims) > 1
  Ms = [randn(xdims[k+1], xdims[k]+1) * σ for k in 1:length(xdims)-1]
  return NeuralNetwork(activ=activ, xdims=xdims, Ms=Ms)
end

