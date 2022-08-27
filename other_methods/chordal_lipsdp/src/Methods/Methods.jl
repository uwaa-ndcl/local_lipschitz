module Methods

include("header.jl");
include("common.jl");
include("lipsdp.jl");
include("chordalsdp.jl");
include("naivelip.jl");
include("cplip.jl");

# Type definitions in core/header.jl
export VecInt, VecF64, MatF64, SpVecInt, SpVecF64, SpMatF64
export Activation, ReluActivation, TanhActivation, NeuralNetwork
export QueryInstance, MethodOptions, QuerySolution

# Common funtionalities in core/common.jl
export e, E, Ec
export makeA, makeB
export γlength, makeT, makeM1, makeM2, makeZ
export makeCliqueInfos

# Method-specific types and polymorphic functions
export LipSdpOptions, ChordalSdpOptions
export NaiveLipOptions, CpLipOptions
export setup!, solve!, runQuery

end
