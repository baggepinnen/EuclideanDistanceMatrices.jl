module EuclideanDistanceMatricesTuringExt

using EuclideanDistanceMatrices, LinearAlgebra, MonteCarloMeasurements
using Turing, Distributions, StatsBase
import EuclideanDistanceMatrices: posterior

function posterior(
    locations::AbstractMatrix{S},
    distances,
    args...;
    tdoa = false,
    nsamples = 3000,
    sampler = NUTS(),
    σL = 0.3,
    σD = 0.3,
) where S
    dim, N = size(locations)
    Nd = length(distances)

    d = if tdoa
        abs.(getindex.(distances, 3))
    else
        (getindex.(distances, 3))
    end

    Turing.@model function model_tdoa(locations, distances, d, ::Type{T} = Float64) where {T}
        P0 ~ isonormal(vec(locations), σL) # These denote the true locations
        P = reshape(P0, dim, N)
        dh = Vector{T}(undef, Nd) # These are the predicted distance measurements
        @views for ind in eachindex(distances)
            (i,j,_) = distances[ind]
            di = norm(P[:,i] - P[:,end]) # Distance from source to i
            dj = norm(P[:,j] - P[:,end]) # Distance from source to j
            dh[ind] = abs(di-dj) # This is the predicted TDOA given the posterior locations
        end
        d ~ isonormal(dh, σD) # Observe TDOAs
    end
    Turing.@model function model(locations, distances, d, ::Type{T} = Float64) where {T}
        P0 ~ isonormal(vec(locations), σL) # These denote the true locations
        P = reshape(P0, dim, N)
        dh = Vector{T}(undef, Nd) # These are the predicted distance measurements
        @views for ind in eachindex(distances)
            (i,j,_) = distances[ind]
            dh[ind] = norm(P[:,i] - P[:,j]) # This is the predicted Euclidean distance given the posterior location
        end
        d ~ isonormal(dh, σD) # Observe distances
    end

    m = (tdoa ? model_tdoa : model)(locations, distances, d)
    # The likelihood is multimodal, the measured locations are used as initial guess
    initial_params = InitFromParams((; P0 = vec(locations)))

    if sampler isa Union{Turing.MAP, Turing.MLE}
        res = if sampler isa Turing.MAP
            maximum_a_posteriori(m, args...; initial_params)
        else
            maximum_likelihood(m, args...; initial_params)
        end
        c = StatsBase.coef(res)
        C = StatsBase.vcov(res)
        names = string.(StatsBase.coefnames(res))

        Pinds = [findfirst(==("P0[$i]"), names) for i in eachindex(locations)]

        Pde = Particles(MvNormal(Vector(c), Symmetric(Matrix(C) + 0.1*minimum(min.(σL,σD))^2*I)))
        P = reshape(Pde[Pinds], dim, N)
        return (P=P, d=[norm(P[:,i] - P[:,j]) for i in 1:N, j in 1:N]), res
    end
    @info "Starting sampling (this might take a while)"
    @time chain = sample(m, sampler, nsamples; initial_params)
    nt = Particles(chain, NamedTuple)
    P = reshape(nt.P0, dim, N)
    nt = (nt..., P=P, d=[norm(P[:,i] - P[:,j]) for i in 1:N, j in 1:N])
    @info "Done"
    return nt, chain
end

isonormal(μ, σ::Real) = MvNormal(μ, abs2(σ) * I)
isonormal(μ, σ::AbstractVector) = MvNormal(μ, Diagonal(abs2.(σ)))

end
