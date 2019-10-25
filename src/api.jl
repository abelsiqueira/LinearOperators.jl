"""
    m, n = size(op)

Return the size of a linear operator as a tuple.
"""
size(op :: ALinOp) = throw(NILO("size"))

"""
    m = size(op, d)

Return the size of a linear operator along dimension `d`.
"""
size(op :: ALinOp, d :: Int) = throw(NILO("size"))

"""
    m, n = shape(op)

An alias for size.
"""
shape(op :: ALinOp) = size(op)

"""
    hermitian(op)
    ishermitian(op)

Determine whether the operator is Hermitian.
Operators which can't be verified will return false.
"""
hermitian(op :: ALinOp) = false
ishermitian(op :: ALinOp) = hermitian(op)

"""
    symmetric(op)
    issymmetric(op)

Determine whether the operator is symmetric.
Operators which can't be verified will return false.
"""
symmetric(op :: ALinOp) = false
issymmetric(op :: ALinOp) = symmetric(op)


"""
    show(io, op)

Display basic information about a linear operator.
"""
function show(io :: IO, op :: ALinOp)
  s  = "Generic print for Linear operator\n"
  print(io, s)
end

"""
    A = Matrix(op)

Materialize an operator as a dense array using compute `op * eᵢ` for each
column of the identity matrix.
"""
function Base.Matrix(op :: ALinOp)
  (m, n) = size(op)
  A = Array{eltype(op)}(undef, m, n)
  ei = zeros(eltype(op), n)
  for i = 1 : n
    ei[i] = 1
    A[:, i] = op * ei
    ei[i] = 0
  end
  return A
end

linop_prod(op :: ALinOp, v :: AbstractVector) = throw(NILO("prod"))

linop_tprod(op :: ALinOp, v :: AbstractVector) = throw(NILO("tprod"))

linop_ctprod(op :: ALinOp, v :: AbstractVector) = throw(NILO("ctprod"))

*(op :: ALinOp, v :: AbstractVector) = linop_prod(op, v)

function mul!(y :: AbstractVector, op :: ALinOp, x :: AbstractVector)
  y .= op * x
  return y
end
