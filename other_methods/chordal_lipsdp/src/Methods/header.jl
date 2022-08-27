using Parameters
using SparseArrays

# Helpful type aliases
const VecInt = Vector{Int}
const VecF64 = Vector{Float64}
const MatF64 = Matrix{Float64}
const SpVecInt = SparseVector{Int, Int}
const SpVecF64 = SparseVector{Float64, Int}
const SpMatF64 = SparseMatrixCSC{Float64, Int}

# Different types of networks
abstract type Activation end
struct ReluActivation <: Activation end
struct TanhActivation <: Activation end

# Generic neural network supertype
@with_kw struct NeuralNetwork
  # The type of the network
  activ::Activation

  # The state vector dimension at start of each layer
  xdims::VecInt

  # The dimensions for the Z (edims) and T (fdims)
  edims::VecInt = xdims[1:end-1]
  fdims::VecInt = edims[2:end]

  # Each M[K] == [Wk bk]
  Ms::Vector{MatF64}
  K::Int = length(Ms)

  # Assert a non-trivial structural integrity of the network
  @assert length(xdims) >= 2
  @assert length(xdims) == K + 1
  @assert all([size(Ms[k]) == (xdims[k+1], xdims[k]+1) for k in 1:K])
end

# Query instance
@with_kw struct QueryInstance
  ffnet::NeuralNetwork
end

# The type of options
abstract type MethodOptions end

# The solution that is to be output by an algorithm
@with_kw struct QuerySolution{A, B, C, D}
  objective_value::A
  values::B
  summary::C
  termination_status::D
  total_time::Float64
  setup_time::Float64
  solve_time::Float64
end

