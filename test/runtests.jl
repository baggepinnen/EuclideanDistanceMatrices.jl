using EuclideanDistanceMatrices
using Test, LinearAlgebra, Statistics
using Distances


@testset "lowrankapprox" begin
    @info "Testing lowrankapprox"
    @test lowrankapprox(randn(10,10), 3) |> rank == 3
    @test lowrankapprox(randn(10,10), 9) |> rank == 9


    P = randn(2,40)
    Pn = P + 0.1 * randn(size(P))
    D = pairwise(SqEuclidean(), P, dims=2)
    Dn = D + 0.1 * randn(size(D))
    Dd = denoise_distmat(Dn, 2, 2)
    @test norm(Dd-D) < norm(Dn-D)
    @test rank(Dd) == 2+2


    Dn = D + (50 * randn(size(D)) .* (rand(size(D)...) .< 0.1))
    Dd = denoise_distmat(Dn, 2, 1)
    @test norm(Dd-D) < norm(Dn-D)


end

@testset "EuclideanDistanceMatrices.jl" begin
    @testset "complete distmat" begin
        @info "Testing complete distmat"

        P = randn(2,40)
        D = pairwise(SqEuclidean(), P, dims=2)
        W = rand(size(D)...) .> 0.3 # Create a random mask
        W = (W + W') .> 0           # It makes sense for the mask to be symmetric
        W[diagind(W)] .= true
        D0 = W .* D                 # Remove missing entries

        D2,S = complete_distmat(D0, W)

        @test (norm(D-D2)/norm(D)) < 1e-5
        @test (norm(W .* (D-D2))/norm(D)) < 1e-5

        X  = reconstruct_pointset(S, 2)

        # Verify that reconstructed `X` is correct up to rotation and translation
        A = [X' ones(size(D,1))]
        P2 = (A*(A \ P'))'
        @test norm(P-P2)/norm(P) < 1e-3

    end
end


@testset "procrustes" begin
    @info "Testing procrustes"

    θ = 1
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    t = randn(2)
    Y = randn(2,40)
    X = R*Y .+ t
    Ri = R'
    ti = -R't

    R2, t2 = procrustes(X,Y)

    @test t2 ≈ ti
    @test R2 ≈ Ri
    @test R2*X .+ t2 ≈ Y

end

@testset "psoterior" begin
    @info "Testing psoterior"

    N = 10

    σL = 0.1
    σD = 0.01

    P = randn(2,N)
    Pn = P + σL*randn(size(P))
    D = pairwise(SqEuclidean(), P, dims=2)
    Dn = D + σD*randn(size(D))
    Dn[diagind(Dn)] .= 0


    distances = []
    p = 0.6
    for i = 1:N
        for j = i+1:N-1
            rand() < p || continue
            push!(distances, (i,j,Dn[i,j]))
        end
    end
    @show length(distances)
    @show expected = p*(N^2-N)÷2


    part, chain = posterior(
        Pn,
        distances;
        nsamples = 1500,
        sampler = NUTS(),
        σL = σL,
        σD = σD
    )

    @test norm(mean.(part.P) - P) < norm(Pn - P)

    if isinteractive()
        scatter(part.P[1,:], part.P[2,:], markersize=6)
        scatter!(P[1,:], P[2,:], lab="True positions")
        scatter!(Pn[1,:], Pn[2,:], lab="Measured positions") |> display
    end



    part, res = posterior(
        Pn,
        distances;
        sampler = MAP(),
        σL = σL,
        σD = σD
    )
    @test norm(mean.(part.P) - P) < norm(Pn - P)

    if isinteractive()
        scatter(part.P[1,:], part.P[2,:], markersize=6)
        scatter!(P[1,:], P[2,:], lab="True positions")
        scatter!(Pn[1,:], Pn[2,:], lab="Measured positions") |> display
    end

end
