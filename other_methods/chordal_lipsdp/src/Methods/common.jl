using LinearAlgebra
using SparseArrays

# Splice a vector
function splice(x, sizes::VecInt)
  @assert all(sizes .>= 0)
  @assert 1 <= length(x) == sum(sizes)
  num_sizes = length(sizes)
  highs = [sum(sizes[1:k]) for k in 1:num_sizes]
  lows = [1; [1 + highk for highk in highs[1:end-1]]]
  @assert length(highs) == length(lows)
  splices = [x[lows[k] : highs[k]] for k in 1:num_sizes]
  return splices
end

# The ith basis vector
function e(i::Int, dim::Int)
  @assert 1 <= i <= dim
  e = spzeros(dim)
  e[i] = 1
  return e
end

# The ith block index matrix
function E(i::Int, dims::VecInt)
  @assert 1 <= i <= length(dims)
  width = sum(dims)
  low = sum(dims[1:i-1]) + 1
  high = sum(dims[1:i])
  E = spzeros(dims[i], width)
  E[1:dims[i], low:high] = I(dims[i])
  return E
end

# Block matrix of [e(k); e(k+1); ...; e(k+size-1)]
# Assertion fails on out-of-bounds
function Ec(k::Int, size::Int, dim::Int)
  @assert k >= 1 && size >= 1
  @assert 1 <= k + size - 1 <= dim
  ets = [e(j, dim)' for j in k:(k+size-1)]
  return vcat(ets...)
end

# Make the A block matrix
function makeA(ffnet::NeuralNetwork)
  Ws = [M[:, 1:end-1] for M in ffnet.Ms]
  A = sum(E(k, ffnet.fdims)' * Ws[k] * E(k, ffnet.edims) for k in 1:(ffnet.K-1))
  return A
end

# Make the B block matrix
function makeB(ffnet::NeuralNetwork)
  B = sum(E(k, ffnet.fdims)' * E(k+1, ffnet.edims) for k in 1:(ffnet.K-1))
  return B
end

# Calculate how long the γ should be given a particular τ
# Note that γ = [γac; γlip]
function γlength(τ::Int, ffnet::NeuralNetwork)
  Tdim = sum(ffnet.fdims)
  @assert 0 <= τ <= Tdim - 1
  return sum((Tdim-τ):Tdim) + 1
end

# The T
function makeT(γac, τ::Int, ffnet::NeuralNetwork)
  @assert length(γac) + 1 == γlength(τ, ffnet)
  Tdim = sum(ffnet.fdims)
  if τ > 0
    ijs = [(i,j) for i in 1:(Tdim-1) for j in (i+1):Tdim if j-i <= τ]
    δts = [e(i,Tdim)' - e(j,Tdim)' for (i,j) in ijs]
    Δ = vcat(δts...)

    # Match the ijs to γac by the above order; offset by Tdim due to the diags
    v = vec([γac[Tdim+ind] for ind in 1:length(ijs)])
    T = Δ' * (v .* Δ)

    # The diagonal components are then added
    T[diagind(T)] .+= γac[1:Tdim]
  else
    T = Diagonal(γac[1:Tdim])
  end
  return T
end

# Construct M1, or smaller variants depending on what is queried with
function makeM1(T, A, B, ffnet::NeuralNetwork)
  @assert ffnet.activ isa ReluActivation || ffnet.activ isa TanhActivation
  low, high = 0, 1
  _Q11 = -2 * low * high * T
  _Q12 = (low + high) * T
  _Q21 = _Q12'
  _Q22 = -2 * T
  Q = [_Q11 _Q12; _Q21 _Q22]
  M1 = [A; B]' * Q * [A; B]
  return M1
end

# Construct M2
function makeM2(γlip, ffnet::NeuralNetwork)
  E1 = E(1, ffnet.edims)
  EK = E(ffnet.K, ffnet.edims)
  WK = ffnet.Ms[ffnet.K][:, 1:end-1]
  _R1 = -γlip * E1' * E1
  _R2 = EK' * (WK' * WK) * EK
  M2 = _R1 + _R2
  return M2
end

# Make the Z
function makeZ(γ, τ, ffnet::NeuralNetwork)
  @assert length(γ) == γlength(τ, ffnet)
  γac, γlip = γ[1:end-1], γ[end]
  T = makeT(γac, τ, ffnet)
  A, B, = makeA(ffnet), makeB(ffnet)
  M1 = makeM1(T, A, B, ffnet)
  M2 = makeM2(γlip, ffnet)
  Z = M1 + M2
  return Z
end

# Calculate the start and size of each clique
function makeCliqueInfos(τ::Int, ffnet::NeuralNetwork)
  edims = ffnet.edims
  Zdim = sum(edims)
  # k, kstart, Ckdim
  clique_infos = Vector{Tuple{Int, Int, Int}}()
  for k in 1:length(edims)
    start = sum(edims[1:k-1])+1
    Ckdim = edims[k] + edims[k+1] + τ
    if start + Ckdim - 1 >= Zdim
      push!(clique_infos, (k, start, Zdim-start+1))
      break
    else
      push!(clique_infos, (k, start, Ckdim))
    end
  end
  return clique_infos
end

