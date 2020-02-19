using Documenter
using LinearOperators

makedocs(
  modules = [LinearOperators],
  doctest = true,
  strict = true,
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "LinearOperators.jl",
  pages = Any["Home" => "index.md",
              "Tutorial" => "tutorial.md",
              "Reference" => "reference.md"]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/LinearOperators.jl.git",
  target = "build",
  devbranch = "master"
)
