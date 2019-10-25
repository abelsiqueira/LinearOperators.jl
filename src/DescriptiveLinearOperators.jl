const FuncOrNothing = Union{Function,Nothing}

mutable struct DescriptiveLinearOperator{T,F1<:FuncOrNothing,F2<:FuncOrNothing,F3<:FuncOrNothing} <: ALinOp{T}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1
  tprod  :: F2
  ctprod :: F3
end

const DescLinOp = DescriptiveLinearOperator

"""
    LinearOperator(nrow, ncol, symmetric, hermitian, prod,
                    [tprod=nothing,
                    ctprod=nothing])

Construct a linear operator from functions.
"""
function LinearOperator(nrow :: Int, ncol :: Int,
                        symmetric :: Bool, hermitian :: Bool,
                        prod :: F1,
                        tprod :: F2=nothing,
                        ctprod :: F3=nothing) where {F1 <: FuncOrNothing,
                                                     F2 <: FuncOrNothing,
                                                     F3 <: FuncOrNothing}
  T = hermitian ? (symmetric ? Float64 : ComplexF64) : ComplexF64
  DescLinOp{T,F1,F2,F3}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end

"""
    LinearOperator(type, nrow, ncol, symmetric, hermitian, prod,
                    [tprod=nothing,
                    ctprod=nothing])

Construct a linear operator from functions where the type is specified as the first argument.
Notice that the linear operator does not enforce the type, so using a wrong type can
result in errors. For instance,
```
A = [im 1.0; 0.0 1.0] # Complex matrix
op = LinearOperator(Float64, 2, 2, false, false, v->A*v, u->transpose(A)*u, w->A'*w)
Matrix(op) # InexactError
```
The error is caused because `Matrix(op)` tries to create a Float64 matrix with the
contents of the complex matrix `A`.
"""
function LinearOperator(::Type{T}, nrow :: Int, ncol :: Int,
                        symmetric :: Bool, hermitian :: Bool,
                        prod :: F1,
                        tprod :: F2=nothing,
                        ctprod :: F3=nothing) where {T,
                                                     F1 <: FuncOrNothing,
                                                     F2 <: FuncOrNothing,
                                                     F3 <: FuncOrNothing}
  DescLinOp{T,F1,F2,F3}(nrow, ncol, symmetric, hermitian, prod, tprod, ctprod)
end

size(op :: DescLinOp) = (op.nrow, op.ncol)

function size(op :: DescLinOp, d :: Int)
  if d == 1
    return op.nrow
  elseif d == 2
    return op.ncol
  end
  throw(LinearOperatorException("Linear operators only have 2 dimensions for now"))
end

hermitian(op :: DescLinOp) = op.hermitian

symmetric(op :: DescLinOp) = op.symmetric

function show(io :: IO, op :: DescLinOp)
  s  = "Generic print for Linear operator\n"
  s *= "  nrow: $(op.nrow)\n"
  s *= "  ncol: $(op.ncol)\n"
  s *= "  eltype: $(eltype(op))\n"
  s *= "  symmetric: $(op.symmetric)\n"
  s *= "  hermitian: $(op.hermitian)\n"
  s *= "\n"
  print(io, s)
end

function linop_prod(op :: DescLinOp, v :: AbstractVector)
  op.ncol == size(v, 1) || throw(LinearOperatorException("shape mismatch"))
  op.prod(v)
end

function linop_tprod(op :: DescLinOp, v :: AbstractVector)
  op.symmetric && return linop_prod(op, v)
  op.tprod !== nothing && return op.tprod(v)
  ctprod = op.ctprod
  if ctprod === nothing
    if op.hermitian
      ctprod = op.prod
    else
      throw(LinearOperatorException("unable to infer tprod"))
    end
  end
  return conj.(ctprod(conj.(v)))
end

function linop_ctprod(op :: DescLinOp, v :: AbstractVector)
  op.hermitian && return linop_prod(op, v)
  op.ctprod !== nothing && return op.ctprod(v)
  tprod = op.tprod
  if tprod === nothing
    if op.symmetric
      tprod = op.prod
    else
      throw(LinearOperatorException("unable to infer ctprod"))
    end
  end
  return conj.(tprod(conj.(v)))
end
