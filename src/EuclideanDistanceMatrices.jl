module EuclideanDistanceMatrices

using LinearAlgebra, Statistics
import Pkg
using StatsBase
using TotalLeastSquares
import Convex
using SCS
using MonteCarloMeasurements
using Turing, Distributions
using Turing2MonteCarloMeasurements


export complete_distmat, rankcomplete_distmat, reconstruct_pointset, denoise_distmat, lowrankapprox, procrustes, posterior, align_to_mean

"""
    D̃, S = complete_distmat(D, W, λ = sqrt(count(W .== 0)); optimizer = SCS.Optimizer)

Takes an incomplete squared Euclidean distance matrix `D` and fills in the missing entries indicated by the mask `W`. `W` is a `BitArray` or array of {0,1} with 0 denoting a missing value. Returns the completed matrix and an SVD object that allows reconstruction of the generating point set `X`.

*NOTE* This function is only available after `using Convex, SCS`, and it will run out of memory for large matrices. See `rankcomplete_distmat` for a less accuraet function that works for larger matrices.

# Arguments:
- `D`: The incomplete matrix
- `W`: The mask
- `λ`: Regularization parameter. A higher value enforces better data fitting, which might be required if the number of entries in `D` is very small.

# Example:
```julia
using EuclideanDistanceMatrices, Distances
P = randn(2,40)
D = pairwise(SqEuclidean(), P)
W = rand(size(D)...) .> 0.3 # Create a random mask
W = (W + W') .> 0           # It makes sense for the mask to be symmetric
W[diagind(W)] .= true
D0 = W .* D                 # Remove missing entries

D2, S = complete_distmat(D0, W)

@show (norm(D-D2)/norm(D))
@show (norm(W .* (D-D2))/norm(D))
```

The set of points that created `D` can be reconstructed up to an arbitrary rotation and translation, `X` contains the reconstruction in the `d` first rows, where `d` is the dimension of the point coordinates. To reconstruct `X` using `S`, do
```julia
X  = reconstruct_pointset(S, 2)

# Verify that reconstructed `X` is correct up to rotation and translation
R,t = procrustes(X,P)
P2 = R*X .+ t
norm(P-P2)/norm(P) # Should be small
```

Ref: Algorithm 5 from "Euclidean Distance Matrices: Essential Theory, Algorithms and Applications"
Ivan Dokmanic, Reza Parhizkar, Juri Ranieri and Martin Vetterli https://arxiv.org/pdf/1502.07541.pdf
"""
function complete_distmat(D, W, λ=sqrt(count(W .== 0)); optimizer = SCS.Optimizer)
    @assert all(==(1), diag(W)) "The diagonal is always observed and equal to 0. Make sure the diagonal of W is true"
    @assert all(iszero, diag(D)) "The diagonal of D is always 0"
    n = size(D, 1)
    x = -1/(n + sqrt(n))
    y = -1/sqrt(n)
    V = [fill(y, 1, n-1); fill(x, n-1,n-1) + I(n-1)]
    e = ones(n)
    G = Convex.Variable((n-1, n-1))
    B = V*G*V'
    E = diag(B)*e' + e*diag(B)' - 2*B
    problem = Convex.maximize(tr(G)- λ * norm(vec(W .* (E - D))), [isposdef(G)])
    Convex.solve!(problem, optimizer)
    Int(problem.status) == 1 || @error problem.status
    B  = Convex.evaluate(B)
    D2 = diag(B)*e' + e*diag(B)' - 2*B
    @info "Data fidelity (norm(W .* (D-D̃))/norm(D))", (norm(W .* (D-D2))/norm(D))
    s  = svd(B)
    D2, s
end


"""
    rankcomplete_distmat(D, W, dim; μ=mean(D[W]), iters=100, tol=1e-6, verbose=true)

Takes an incomplete squared Euclidean distance matrix `D` and fills in the missing entries indicated by the mask `W::BitArray` with 0 denoting a missing value. Returns the completed matrix and an `Eigen` object that allows reconstruction of the generating point set `X`.

This function works for larger matrices than `complete_distmat`, but is much less accurate. See `complete_distmat` for examples.

# Arguments:
- `D`: The incomplete matrix
- `W`: The mask
- `dim`: The dimension of the points that created `D`
- `μ`: Initial value for unobserved entries.
- `iters`: Maximum number of iterations
- `tol`: Tolerance for the change in `D` between iterations.
- `verbose`: print status
"""
function rankcomplete_distmat(D, W, dim; μ=mean(D[W]), iters=100, tol=1e-6, verbose=true)
    @assert all(==(1), diag(W)) "The diagonal is always observed and equal to 0. Make sure the diagonal of W is true"
    @assert all(iszero, diag(D)) "The diagonal of D is always 0"
    n    = size(D, 1)
    D2   = copy(D)
    Dold = copy(D2)
    D2[.!W] .= μ
    local E
    for iter = 1:iters
        D2, E = eigval_th(Symmetric(D2), dim+2)
        D2[W] .= D[W]
        D2[diagind(D2)] .= 0
        D2 .= max.(D2, 0)
        err = sqrt(sum(abs2(d1-d2) for (d1,d2) in zip(Dold, D2)))/norm(D2)
        verbose && @info "Iter $iter Change: $err"
        err < tol && break
        Dold .= D2
    end
    D2, E
end

function eigval_th(D, r)
    E = eigen(D)
    E.values[1:end-r+1] .= 0
    E.vectors*Diagonal(E.values)*E.vectors', E
end

"""
    reconstruct_pointset(D, dim)

Takes a squared distance matrix or the SVD of one and reconstructs the set of points embedded in dimension `dim` that generated `D; up to a translation and rotation/reflection. See `procrustes` for help with aligning the result to a collection of anchors.

The algorithm used if `D` is a distance matrix is called Multidimensional Scaling (MDS) https://en.wikipedia.org/wiki/Multidimensional_scaling
"""
function reconstruct_pointset(D::AbstractMatrix, dim)
    n = size(D,1)
    J = I - fill(1/n, n, n)
    G = -1/2 * J*D*J
    E = eigen(Symmetric(G))
    reconstruct_pointset(E, dim)
end

function reconstruct_pointset(E::Eigen, dim)
    Diagonal(sqrt.(max.(E.values[end-dim+1:end], 0)))*E.vectors[:, end-dim+1:end]'
end

function reconstruct_pointset(S::SVD,dim)
    X  = Diagonal(sqrt.(S.S[1:dim]))*S.Vt[1:dim, :]
end


"""
    denoise_distmat(D, dim, p = 2)

Takes a noisy squared distance matrix and returns a denoised version. `p` denotes the "norm" used in measuring the error. `p=2` assumes that the error is Gaussian, whereas `p=1` assumes that the error is large but sparse.

# Arguments:
- `dim`: The dimension of the points that generated `D`
"""
function denoise_distmat(D, dim, p=2)
    if p == 2
        s = svd(D)
        return lowrankapprox(s, dim+2)
    elseif p == 1
        A,E,s,sv = rpca(D, nonnegA=true)
        return lowrankapprox(s, dim+2)
    else
        throw(ArgumentError("p must be 1 or 2"))
    end
end

function lowrankapprox(s::SVD, r)
    @views s.U[:,1:r] * Diagonal(s.S[1:r]) * s.Vt[1:r,:]
end
lowrankapprox(D, r) = D = lowrankapprox(svd(D), r)


"""
    R,t = procrustes(X, Y)

Find rotation matrix `R` and translation vector `t` such that `R*X .+ t ≈ Y`
"""
function procrustes(X,Y)
    mX = mean(X, dims=2)
    mY = mean(Y, dims=2)
    Xb,Yb = X .- mX, Y .- mY
    s = svd(Xb*Yb')
    R = s.V*s.U'
    R, mY - R*mX
end

"""
    part, result = posterior(locations::AbstractMatrix, distances, args...; nsamples = 3000, sampler = NUTS(), σL = 0.3, σD = 0.3)

Compute the full Bayesian posterior over locations given noisy measurements of both locations and distances.
Missing values can be indicated by using either the `missing` element, or by providing a rough estimate and setting a very large standard deviation for the guessed element.

# Arguments:
- `locations`: A matrix with one point in each column.
- `distances`: A vector of tuples `(i,j,d)` where `i,j` denotes the indices between the distance measured is `d` (not squared).
- `args`: If any provided, they are sent to `optimize`. Can be used to supply a choice of optimizer of `MAP` estimation.
- `nsamples`: How many samples to draw for MCMC sampling.
- `sampler`: Any of the samplers or `MAP/MLE` supported by Turing
- `σL`: The noise std in the locations.
- `σD`: The noise std in the distances.
- `tdoa`: Indicates whether or not `distances` contains TDOA measurements. See note below.

`part` contains `part.P` and `part.d` with the posterior over locations and distances.

## TDOA
When using `tdoa=true`, `distances` is expected to contain tuples `(i,j,tdoa)`, and the last column in `locations` is expected to contain an estimate of the source location. Note that the standard deviation for locations can be a vector, aiding in situations when this location is known with more or less accuracy than the sensors.

# Example
```julia
N = 10

σL = 0.1
σD = 0.01

P = randn(2,N)
Pn = P + σL*randn(size(P))
D = pairwise(Euclidean(), P, dims=2)
Dn = D + σD*randn(size(D))
Dn[diagind(Dn)] .= 0

distances = []
noisy_distances = []
p = 0.5
for i = 1:N
    for j = i+1:N
        if rand() < p
            push!(distances, (i,j,D[i,j]))
            push!(noisy_distances, (i,j,Dn[i,j]))
        end
    end
end
@show length(distances)
@show expected = p*((N^2-N)÷2)

part, chain = posterior(
    Pn,
    noisy_distances;
    nsamples = 2000,
    sampler = NUTS(),
    σL = σL,
    σD = σD
)

norm(pmean.(part.P) - P) < norm(Pn - P)

scatter(part.P[1,:], part.P[2,:], markersize=6, layout=2, sp=1)
scatter!(P[1,:], P[2,:], lab="True positions", sp=1)
scatter!(Pn[1,:], Pn[2,:], lab="Measured positions", sp=1)
bar!(getindex.(distances, 3), sp=2, lab="True dist")
bar!(getindex.(noisy_distances, 3), sp=2, lab="Measured dist", alpha=0.7)
scatter!(part.d, sp=2, seriestype=:scatter) |> display
```
"""
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
        P0 ~ MvNormal(vec(locations), σL) # These denote the true locations
        P = reshape(P0, dim, N)
        dh = Vector{T}(undef, Nd) # These are the predicted distance measurements
        for ind in eachindex(distances)
            (i,j,_) = distances[ind]
            di = norm(P[:,i] - P[:,end]) # Distance from source to i
            dj = norm(P[:,j] - P[:,end]) # Distance from source to j
            dh[ind] = abs(di-dj) # This is the predicted TDOA given the posterior locations
        end
        d ~ MvNormal(dh, σD) # Observe TDOAs
    end
    Turing.@model function model(locations, distances, d, ::Type{T} = Float64) where {T}
        P0 ~ MvNormal(vec(locations), σL) # These denote the true locations
        P = reshape(P0, dim, N)
        dh = Vector{T}(undef, Nd) # These are the predicted distance measurements
        for ind in eachindex(distances)
            (i,j,_) = distances[ind]
            dh[ind] = norm(P[:,i] - P[:,j]) # This is the predicted Euclidean distance given the posterior location
        end
        d ~ MvNormal(dh, σD) # Observe distances
    end

    m = (tdoa ? model : model_tdoa)(locations, distances, d)

    if sampler isa Turing.Inference.InferenceAlgorithm
        @info "Starting sampling (this might take a while)"
        @time chain = sample(m, sampler, nsamples, drop_warmup=true)
        nt = Particles(chain)
        P = reshape(nt.P0, dim, N)
        nt = (nt..., P=P, d=[norm(P[:,i] - P[:,j]) for i in 1:N, j in 1:N])
        @info "Done"
        return nt, chain
    elseif typeof(sampler) ∈ (Turing.MAP, Turing.MLE)
        if sampler isa Turing.MAP
            res = maximum_a_posteriori(m, args...)
        else
            res = maximum_likelihood(m, args...)
        end
        c = StatsBase.coef(res)
        C = StatsBase.vcov(res)
        names = StatsBase.params(res)

        Pinds = findfirst.([==(Symbol("P0[$i]")) for i in eachindex(locations)], Ref(names))
        # dinds = findfirst.([==("d[$i]") for i in eachindex(distances)], Ref(names))

        Pde = Particles(MvNormal(Vector(c), Symmetric(Matrix(C) + 0.1*min(σL,σD)^2*I)))
        P = reshape(Pde[Pinds], dim, N)
        # de = Pde[dinds]
        (P=P, d=[norm(P[:,i] - P[:,j]) for i in 1:N, j in 1:N]), res
    end
    # chain
end


"""
    align_to_mean(P)

Takes an array of particles `P` and aligns all samples to the mean array using `procrustes`. This takes `P` from an estimate of absolute quantities to an estimate of relative quantities, which, in general, will exhibit lower variance.
"""
function align_to_mean(P)
    m = pmean.(P)
    function apply_procrustes(X)
        R, t = procrustes(X, m)
        R * X .+ t
    end
    aligned_P = ℝⁿ2ℝⁿ_function(apply_procrustes, P)
end


end
