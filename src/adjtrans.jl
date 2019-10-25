export AdjointLinearOperator, TransposeLinearOperator, adjoint, transpose

# From julialang:stdlib/LinearAlgebra/src/adjtrans.jl
struct AdjointLinearOperator{T,S} <: ALinOp{T}
  parent :: S
  function AdjointLinearOperator{T,S}(op :: S) where {T,S}
    @assert eltype(op) == T
    new(op)
  end
end

struct TransposeLinearOperator{T,S} <: ALinOp{T}
  parent :: ALinOp{T}
  function TransposeLinearOperator{T,S}(op :: S) where {T,S}
    @assert eltype(op) == T
    new(op)
  end
end

struct ConjugateLinearOperator{T,S} <: ALinOp{T}
  parent :: ALinOp{T}
  function ConjugateLinearOperator{T,S}(op :: S) where {T,S}
    @assert eltype(op) == T
    T <: Real && return op
    new(op)
  end
end

AdjointLinearOperator(A)   = AdjointLinearOperator{eltype(A),typeof(A)}(A)
TransposeLinearOperator(A) = TransposeLinearOperator{eltype(A),typeof(A)}(A)
ConjugateLinearOperator(A) = ConjugateLinearOperator{eltype(A),typeof(A)}(A)

adjoint(A :: ALinOp) = AdjointLinearOperator(A)
transpose(A :: ALinOp) = TransposeLinearOperator(A)
conj(A :: ALinOp) = ConjugateLinearOperator(A)

adjoint(A :: AdjointLinearOperator) = A.parent
transpose(A :: TransposeLinearOperator) = A.parent
conj(A :: ConjugateLinearOperator) = A.parent

adjoint(A :: ConjugateLinearOperator) = transpose(A.parent)
adjoint(A :: TransposeLinearOperator) = conj(A.parent)
conj(A :: AdjointLinearOperator) = transpose(A.parent)
conj(A :: TransposeLinearOperator) = adjoint(A.parent)
transpose(A :: AdjointLinearOperator) = conj(A.parent)
transpose(A :: ConjugateLinearOperator) = adjoint(A.parent)

const AdjTrans = Union{AdjointLinearOperator,TransposeLinearOperator}

size(A :: AdjTrans) = size(A.parent)[[2;1]]
size(A :: AdjTrans, d :: Int) = size(A.parent, 3 - d)
size(A :: ConjugateLinearOperator) = size(A.parent)
size(A :: ConjugateLinearOperator, d :: Int) = size(A.parent, d)

for f in [:hermitian, :ishermitian, :symmetric, :issymmetric]
  @eval begin
    $f(A :: AdjTrans) = $f(A.parent)
  end
end

function show(io :: IO, op :: AdjointLinearOperator)
  println(io, "Adjoint of the following LinearOperator:")
  show(io, op.parent)
end

function show(io :: IO, op :: TransposeLinearOperator)
  println(io, "Transpose of the following LinearOperator:")
  show(io, op.parent)
end

function show(io :: IO, op :: ConjugateLinearOperator)
  println(io, "Conjugate of the following LinearOperator:")
  show(io, op.parent)
end

linop_prod(op :: TransposeLinearOperator, v :: AbstractVector) = linop_tprod(op.parent, v)
linop_prod(op :: AdjointLinearOperator, v :: AbstractVector) = linop_ctprod(op.parent, v)

function linop_prod(op :: ConjugateLinearOperator, v :: AbstractVector)
  p = op.parent
  return conj.(p * v)
end
