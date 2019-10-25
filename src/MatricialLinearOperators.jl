mutable struct MatricialLinearOperator{T} <: ALinOp{T}
  matrix :: Matrix{T}
end

const MatLinOp = MatricialLinearOperator

"""
    LinearOperator(M)

Construct a linear operator from a dense or sparse matrix.
"""
function LinearOperator(M :: AbstractMatrix{T}) where T
  MatricialLinearOperator{T}(M)
end

size(op :: MatLinOp) = size(op.matrix)

size(op :: MatLinOp, d :: Int) = size(op.matrix, d)

hermitian(op :: MatLinOp) = ishermitian(op.matrix)

symmetric(op :: MatLinOp) = issymmetric(op.matrix)

function show(io :: IO, op :: MatLinOp)
  println(io, "Matricial Linear operator")
  show(io, op.matrix)
  println(io, "")
end

linop_prod(op :: MatLinOp, v :: AbstractVector) = op.matrix * v

linop_tprod(op :: MatLinOp, v :: AbstractVector) = transpose(op.matrix) * v

linop_ctprod(op :: MatLinOp, v :: AbstractVector) = op.matrix' * v

Base.Matrix(op :: MatLinOp) = op.matrix

mul!(y :: AbstractVector, op :: MatLinOp, x :: AbstractVector) = mul!(y, op.matrix, x)
