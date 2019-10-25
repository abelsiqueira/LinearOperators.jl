# Linear Operators for Julia
module LinearOperators

using FastClosures, Printf, LinearAlgebra, SparseArrays

export AbstractLinearOperator, LinearOperator,
       NotImplementedLinearOperator,
       LinearOperatorException, mul!,
       opEye, opOnes, opZeros, opDiagonal,
       opInverse, opCholesky, opLDL, opHouseholder, opHermitian,
       check_ctranspose, check_hermitian, check_positive_definite,
       shape, hermitian, ishermitian, symmetric, issymmetric,
       opRestriction, opExtension


mutable struct LinearOperatorException <: Exception
  msg :: AbstractString
end

# when indexing, Colon() is treated separately
const LinearOperatorIndexType = Union{UnitRange{Int}, StepRange{Int, Int}, AbstractVector{Int}}

# import methods we overload
import Base.eltype, Base.isreal, Base.size, Base.show
import Base.+, Base.-, Base.*
import Base.transpose
import Base.adjoint
import LinearAlgebra.issymmetric, LinearAlgebra.ishermitian, LinearAlgebra.mul!
import Base.conj
import Base.hcat, Base.vcat, Base.hvcat

"""
Base abstract type to represent a linear operator.
The usual arithmetic operations may be applied to operators
to combine or otherwise alter them. They can be combined with
other operators, with matrices and with scalars. Some Operators may
be transposed and conjugate-transposed using the usual Julia syntax.
"""
abstract type AbstractLinearOperator{T} end

const ALinOp = AbstractLinearOperator
const OperatorOrMatrix = Union{AbstractLinearOperator, AbstractMatrix}

eltype(A :: AbstractLinearOperator{T}) where {T} = T
isreal(A :: AbstractLinearOperator{T}) where {T} = T <: Real

mutable struct NotImplementedLinearOperator <: Exception
  name :: Union{Symbol,Function,String}
end
const NILO = NotImplementedLinearOperator

Base.showerror(io::IO, e::NILO) = print(io, e.name, " not implemented for this LinearOperator type")

# Basic definitions and expected API for specific linear operators
include("api.jl")
+(op :: ALinOp) = op

# Utility functions
include("utility.jl")

# Linear operator types
include("adjtrans.jl")
include("DescriptiveLinearOperators.jl")
include("MatricialLinearOperators.jl")

#include("PreallocatedLinearOperators.jl")

#=
# Apply an operator to a vector.
function *(op :: AbstractLinearOperator, v :: AbstractVector)
  size(v, 1) == size(op, 2) || throw(LinearOperatorException("shape mismatch"))
  op.prod(v)
end

function -(op :: AbstractLinearOperator{T,F1,F2,F3}) where {T,F1,F2,F3}
  prod = @closure v -> -op.prod(v)
  tprod = @closure u -> -op.tprod(u)
  ctprod = @closure w -> -op.ctprod(w)
  F4 = typeof(prod)
  F5 = typeof(tprod)
  F6 = typeof(ctprod)
  LinearOperator{T,F4,F5,F6}(op.nrow, op.ncol, op.symmetric, op.hermitian, prod, tprod, ctprod)
end

# Binary operations.

## Operator times operator.
function *(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  prod = @closure v -> op1 * (op2 * v)
  tprod = @closure u -> transpose(op2) * (transpose(op1) * u)
  ctprod = @closure w -> op2' * (op1' * w)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  LinearOperator{S,F1,F2,F3}(m1, n2, false, false, prod, tprod, ctprod)
end

## Matrix times operator.
*(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) * op
*(op :: AbstractLinearOperator, M :: AbstractMatrix) = op * LinearOperator(M)

## Scalar times operator.
function *(op :: AbstractLinearOperator, x :: Number)
  S = promote_type(eltype(op), typeof(x))
  prod = @closure v -> (op * v) * x
  tprod = @closure u -> x * (transpose(op) * u)
  ctprod = @closure w -> x' * (op' * w)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  LinearOperator{S,F1,F2,F3}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x), prod, tprod, ctprod)
end

function *(x :: Number, op :: AbstractLinearOperator)
  S = promote_type(eltype(op), typeof(x))
  prod = @closure v -> x * (op * v)
  tprod = @closure u -> (transpose(op) * u) * x
  ctprod = @closure w -> (op' * w) * x'
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  LinearOperator{S,F1,F2,F3}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x), prod, tprod, ctprod)
end

# Operator + operator.
function +(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  prod = @closure v -> (op1   * v) + (op2   * v)
  tprod = @closure u -> (transpose(op1) * u) + (transpose(op2) * u)
  ctprod = @closure w -> (op1'  * w) + (op2'  * w)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{S,F1,F2,F3}(m1, n1, op1.symmetric && op2.symmetric, op1.hermitian && op2.hermitian,
                                    prod, tprod, ctprod)
end

# Operator + matrix.
+(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) + op
+(op :: AbstractLinearOperator, M :: AbstractMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
+(op :: AbstractLinearOperator, x :: Number) = op + x * opOnes(op.nrow, op.ncol)
+(x :: Number, op :: AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
-(op1 :: AbstractLinearOperator, op2 :: AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M :: AbstractMatrix, op :: AbstractLinearOperator) = LinearOperator(M) - op
-(op :: AbstractLinearOperator, M :: AbstractMatrix) = op - LinearOperator(M)

# Operator - scalar.
-(op :: AbstractLinearOperator, x :: Number) = op + (-x)
-(x :: Number, op :: AbstractLinearOperator) = x + (-op)
=#



#=
# Special linear operators.

"""`opEye()`

Identity operator.
```
opI = opEye()
v = rand(5)
@assert opI * v === v
```
"""
struct opEye <: AbstractLinearOperator{Any,Nothing,Nothing,Nothing} end

*(::opEye, x :: AbstractArray{T,1} where T) = x
*(x :: AbstractArray{T,1} where T, ::opEye) = x
*(::opEye, A :: AbstractArray{T,2} where T) = A
*(A :: AbstractArray{T,2} where T, ::opEye) = A
*(::opEye, T :: AbstractLinearOperator) = T
*(T :: AbstractLinearOperator, ::opEye) = T
*(::opEye, T::opEye) = T

function show(io :: IO, op :: opEye)
  println(io, "Identity operator")
end

"""
    opEye(T, n)
    opEye(n)

Identity operator of order `n` and of data type `T` (defaults to `Float64`).
"""
function opEye(T :: DataType, n :: Int)
  prod = @closure v -> copy(v)
  F = typeof(prod)
  LinearOperator{T,F,F,F}(n, n, true, true, prod, prod, prod)
end

opEye(n :: Int) = opEye(Float64, n)

# TODO: not type stable
"""
    opEye(T, nrow, ncol)
    opEye(nrow, ncol)

Rectangular identity operator of size `nrow`x`ncol` and of data type `T`
(defaults to `Float64`).
"""
function opEye(T :: DataType, nrow :: Int, ncol :: Int)
  if nrow == ncol
    return opEye(T, nrow)
  end
  if nrow > ncol
    prod = @closure v -> [v ; zeros(T, nrow - ncol)]
    tprod = @closure v -> v[1:ncol]
  else
    prod = @closure v -> v[1:nrow]
    tprod = @closure v -> [v ; zeros(T, ncol - nrow)]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  return LinearOperator{T,F1,F2,F2}(nrow, ncol, false, false, prod, tprod, tprod)
end

opEye(nrow :: Int, ncol :: Int) = opEye(Float64, nrow, ncol)

"""
    opOnes(T, nrow, ncol)
    opOnes(nrow, ncol)

Operator of all ones of size `nrow`-by-`ncol` and of data type `T` (defaults to
`Float64`).
"""
function opOnes(T :: DataType, nrow :: Int, ncol :: Int)
  prod = @closure v -> sum(v) * ones(T, nrow)
  tprod = @closure u -> sum(u) * ones(T, ncol)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  LinearOperator{T,F1,F2,F2}(nrow, ncol, nrow == ncol, nrow == ncol, prod, tprod, tprod)
end

opOnes(nrow :: Int, ncol :: Int) = opOnes(Float64, nrow, ncol)

"""
    opZeros(T, nrow, ncol)
    opZeros(nrow, ncol)

Zero operator of size `nrow`-by-`ncol` and of data type `T` (defaults to
`Float64`).
"""
function opZeros(T :: DataType, nrow :: Int, ncol :: Int)
  prod = @closure v -> zeros(T, nrow)
  tprod = @closure u -> zeros(T, ncol)
  F1 = typeof(prod)
  F2 = typeof(tprod)
  LinearOperator{T,F1,F2,F2}(nrow, ncol, nrow == ncol, nrow == ncol, prod, tprod, tprod)
end

opZeros(nrow :: Int, ncol :: Int) = opZeros(Float64, nrow, ncol)

"""
    opDiagonal(d)

Diagonal operator with the vector `d` on its main diagonal.
"""
function opDiagonal(d :: AbstractVector{T}) where T
  prod = @closure v -> v .* d
  ctprod = @closure w -> w .* conj(d)
  F1 = typeof(prod)
  F2 = typeof(ctprod)
  LinearOperator{T,F1,F1,F2}(length(d), length(d), true, isreal(d),
                             prod,
                             prod,
                             ctprod)
end

#TODO: not type stable
"""
    opDiagonal(nrow, ncol, d)

Rectangular diagonal operator of size `nrow`-by-`ncol` with the vector `d` on
its main diagonal.
"""
function opDiagonal(nrow :: Int, ncol :: Int, d :: AbstractVector{T}) where T
  nrow == ncol <= length(d) && (return opDiagonal(d[1:nrow]))
  if nrow > ncol
    prod = @closure v -> [v .* d ; zeros(nrow-ncol)]
    tprod = @closure u -> u[1:ncol] .* d
    ctprod = @closure w -> w[1:ncol] .* conj(d)
  else
    prod = @closure v -> v[1:nrow] .* d
    tprod = @closure u -> [u .* d ; zeros(ncol-nrow)]
    ctprod = @closure w -> [w .* conj(d) ; zeros(ncol-nrow)]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  LinearOperator{T,F1,F2,F3}(nrow, ncol, false, false, prod, tprod, ctprod)
end


hcat(A :: AbstractLinearOperator, B :: AbstractMatrix) = hcat(A, LinearOperator(B))

hcat(A :: AbstractMatrix, B :: AbstractLinearOperator) = hcat(LinearOperator(A), B)

function hcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  A.nrow == B.nrow || throw(LinearOperatorException("hcat: inconsistent row sizes"))

  nrow  = A.nrow
  ncol  = A.ncol + B.ncol
  S = promote_type(eltype(A), eltype(B))

  prod = @closure v -> A * v[1:A.ncol] + B * v[A.ncol+1:length(v)]
  tprod  = @closure v -> [transpose(A) * v; transpose(B) * v;]
  ctprod = @closure v -> [A' * v; B' * v;]
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  LinearOperator{S,F1,F2,F3}(nrow, ncol, false, false, prod, tprod, ctprod)
end

function hcat(ops :: OperatorOrMatrix...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op ops[i]]
  end
  return op
end


vcat(A :: AbstractLinearOperator, B :: AbstractMatrix) = vcat(A, LinearOperator(B))

vcat(A :: AbstractMatrix, B :: AbstractLinearOperator) = vcat(LinearOperator(A), B)

function vcat(A :: AbstractLinearOperator, B :: AbstractLinearOperator)
  A.ncol == B.ncol || throw(LinearOperatorException("vcat: inconsistent column sizes"))

  nrow  = A.nrow + B.nrow
  ncol  = A.ncol
  S = promote_type(eltype(A), eltype(B))

  prod = @closure v -> [A * v; B * v;]
  tprod = @closure v -> transpose(A) * v +  transpose(B) * v
  ctprod = @closure v -> A' * v[1:A.nrow] + B' * v[A.nrow+1:length(v)]
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{S,F1,F2,F3}(nrow, ncol, false, false, prod, tprod, ctprod)
end

function vcat(ops :: OperatorOrMatrix...)
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end

# Removed by https://github.com/JuliaLang/julia/pull/24017
function hvcat(rows :: Tuple{Vararg{Int}}, ops :: OperatorOrMatrix...)
  nbr = length(rows)
  rs = Array{OperatorOrMatrix,1}(undef, nbr)
  a = 1
  for i = 1:nbr
    rs[i] = hcat(ops[a:a-1+rows[i]]...)
    a += rows[i]
  end
  vcat(rs...)
end

"""
    opInverse(M; symmetric=false, hermitian=false)

Inverse of a matrix as a linear operator using `\\`.
Useful for triangular matrices. Note that each application of this
operator applies `\\`.
"""
function opInverse(M :: AbstractMatrix{T}; symmetric=false, hermitian=false) where T
  prod = @closure v -> M \ v
  tprod = @closure u -> transpose(M) \ u
  ctprod = @closure w -> M' \ w
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  LinearOperator{T,F1,F2,F3}(size(M,2), size(M,1), symmetric, hermitian, prod, tprod, ctprod)
end

"""
    opCholesky(M, [check=false])

Inverse of a Hermitian and positive definite matrix as a linear operator
using its Cholesky factorization. The factorization is computed only once.
The optional `check` argument will perform cheap hermicity and definiteness
checks.
"""
function opCholesky(M :: AbstractMatrix; check :: Bool=false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
    check_positive_definite(M) || throw(LinearOperatorException("matrix is not positive definite"))
  end
  LL = cholesky(M)
  prod = @closure v -> LL \ v
  tprod = @closure u -> conj(LL \ conj(u))  # M.' = conj(M)
  ctprod = @closure w -> LL \ w
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  S = eltype(LL)
  LinearOperator{S,F1,F2,F3}(m, m, isreal(M), true, prod, tprod, ctprod)
  #TODO: use iterative refinement.
end

"""
    opLDL(M, [check=false])

Inverse of a symmetric matrix as a linear operator using its LDL' factorization
if it exists. The factorization is computed only once. The optional `check`
argument will perform a cheap hermicity check.
"""
function opLDL(M :: AbstractMatrix; check :: Bool=false)
  (m, n) = size(M)
  m == n || throw(LinearOperatorException("shape mismatch"))
  if check
    check_hermitian(M) || throw(LinearOperatorException("matrix is not Hermitian"))
  end
  LDL = ldlt(M)
  prod = @closure v -> LDL \ v
  tprod = @closure u -> conj(LDL \ conj(u))  # M.' = conj(M)
  ctprod = @closure w -> LDL \ w
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  S = eltype(LDL)
  return LinearOperator{S,F1,F2,F3}(m, m, isreal(M), true, prod, tprod, ctprod)
  #TODO: use iterative refinement.
end

"""
    opHouseholder(h)

Apply a Householder transformation defined by the vector `h`.
The result is `x -> (I - 2 h h') x`.
"""
function opHouseholder(h :: AbstractVector{T}) where T
  n = length(h)
  prod = @closure v -> (v - 2 * dot(h, v) * h)  # tprod will be inferred
  F1 = typeof(prod)
  LinearOperator{T,F1,Nothing,F1}(n, n, isreal(h), true, prod, nothing, prod)
end

"""
    opHermitian(d, A)

A symmetric/hermitian operator based on the diagonal `d` and lower triangle of `A`.
"""
function opHermitian(d :: AbstractVector{S}, A :: AbstractMatrix{T}) where {S, T}
  m, n = size(A)
  m == n == length(d) || throw(LinearOperatorException("shape mismatch"))
  L = tril(A, -1)
  U = promote_type(S, T)
  prod = @closure v -> (d .* v + L * v + (v' * L)')[:]
  F = typeof(prod)
  LinearOperator{U,F,Nothing,Nothing}(m, m, isreal(A), true, prod, nothing, nothing)
end


"""
    opHermitian(A)

A symmetric/hermitian operator based on a matrix.
"""
function opHermitian(T :: AbstractMatrix)
  d = diag(T)
  opHermitian(d, T)
end

include("qn.jl")  # quasi-Newton operators
include("kron.jl")

"""
    Z = opRestriction(I, ncol)
    Z = opRestriction(:, ncol)

Creates a LinearOperator restricting a `ncol`-sized vector to indices `I`.
The operation `Z * v` is equivalent to `v[I]`. `I` can be `:`.

    Z = opRestriction(k, ncol)

Alias for `opRestriction([k], ncol)`.
"""
function opRestriction(I :: LinearOperatorIndexType, ncol :: Int)
  all(1 .≤ I .≤ ncol) || throw(LinearOperatorException("indices should be between 1 and $ncol"))
  nrow = length(I)
  prod = @closure x -> x[I]
  tprod = @closure x -> begin
    z = zeros(eltype(x), ncol)
    z[I] = x
    return z
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  return LinearOperator{Int,F1,F2,F2}(nrow, ncol, false, false, prod, tprod, tprod)
end

opRestriction(::Colon, ncol :: Int) = opEye(Int, ncol)

opRestriction(k :: Int, ncol :: Int) = opRestriction([k], ncol)

"""
    Z = opExtension(I, ncol)
    Z = opExtension(:, ncol)

Creates a LinearOperator extending a vector of size `length(I)` to size `ncol`,
where the position of the elements on the new vector are given by the indices
`I`.
The operation `w = Z * v` is equivalent to `w = zeros(ncol); w[I] = v`.

    Z = opExtension(k, ncol)

Alias for `opExtension([k], ncol)`.
"""
opExtension(I :: LinearOperatorIndexType, ncol :: Int) = opRestriction(I, ncol)'

opExtension(::Colon, ncol :: Int) = opEye(Int, ncol)

opExtension(k :: Int, ncol :: Int) = opExtension([k], ncol)

# indexing for linear operators
import Base.getindex
function getindex(op :: AbstractLinearOperator,
                  rows :: Union{LinearOperatorIndexType, Int, Colon},
                  cols :: Union{LinearOperatorIndexType, Int, Colon})
  R = opRestriction(rows, size(op, 1))
  E = opExtension(cols, size(op, 2))
  return R * op * E
end
=#

end  # module
