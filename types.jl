
try
     type LRModel
          N::Int
          D::Int
          C::Int
          M::Int
          V0::Float64
          X::Array{Float64,2}
          y::Array{Float64,1}
          y_bin::Array{Float64,2}
          mN::Array{Float64,2}
     end
catch
     # already defined...
end

try
     type MALRModel
          N::Int
          D::Int
          C::Int
          M::Int
          R::Int
          V0::Float64
          tau::Float64
          X::Array{Float64,2}
          y::Array{Float64,1}
          Y::Array{Float64,2}
          mN::Array{Float64,2}
          pi
          latent_posterior
     end
catch
     # already defined...
end
