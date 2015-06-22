ϵ = eps(Float64)
rtol = sqrt(ϵ)

n = 10
mem = 5
B = LBFGSOperator(n, mem)
H = InverseLBFGSOperator(n, mem)

@assert B.data.insert == 1
@assert H.data.insert == 1
@test norm(full(B) - eye(n)) <= ϵ
@test norm(full(H) - eye(n)) <= ϵ

# Test that nonpositive curvature can't be added.
s = rand(n)
z = zeros(n)
push!(B, s, -s); @assert B.data.insert == 1
push!(B, s,  z); @assert B.data.insert == 1
push!(H, s, -s); @assert H.data.insert == 1
push!(H, s,  z); @assert H.data.insert == 1

# Insert a few {s,y} pairs.
for i = 1 : mem+2
  s = rand(n)
  y = rand(n)
  if dot(s, y) >= 1.0e-20
    push!(B, s, y)
    push!(H, s, y)
  end
end

@assert B.data.insert == 3
@assert H.data.insert == 3

@test check_positive_definite(B)
@test check_positive_definite(H)

@test check_hermitian(B)
@test check_hermitian(H)

@test norm(full(H*B) - eye(n)) <= rtol
