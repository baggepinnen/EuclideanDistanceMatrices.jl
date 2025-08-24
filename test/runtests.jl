using EuclideanDistanceMatrices
using Test, LinearAlgebra, Statistics
using Distances, Turing, Optim, MonteCarloMeasurements

function apply_procrustes(X, Y)
    R, t = procrustes(X, Y)
    R * X .+ t
end
@testset "EuclideanDistanceMatrices.jl" begin

    @testset "lowrankapprox" begin
        @info "Testing lowrankapprox"
        @test lowrankapprox(randn(10, 10), 3) |> rank == 3
        @test lowrankapprox(randn(10, 10), 9) |> rank == 9


        P = randn(2, 40)
        Pn = P + 0.1 * randn(size(P))
        D = pairwise(SqEuclidean(), P, dims = 2)
        Dn = D + 0.1 * randn(size(D))
        Dd = denoise_distmat(Dn, 2, 2)
        @test norm(Dd - D) < norm(Dn - D)
        @test rank(Dd) == 2 + 2


        Dn = D + (50 * randn(size(D)) .* (rand(size(D)...) .< 0.1))
        Dd = denoise_distmat(Dn, 2, 1)
        @test norm(Dd - D) < norm(Dn - D)

        @test_throws ArgumentError denoise_distmat(Dn, 2, 3)
    end

    @testset "complete distmat" begin
        @info "Testing complete distmat"

        P = randn(2, 40)
        D = pairwise(SqEuclidean(), P, dims = 2)
        W = rand(size(D)...) .> 0.3 # Create a random mask
        W = (W + W') .> 0           # It makes sense for the mask to be symmetric
        W[diagind(W)] .= true
        D0 = W .* D                 # Remove missing entries

        D2, S = complete_distmat(D0, W)

        @test (norm(D - D2) / norm(D)) < 2e-4
        @test (norm(W .* (D - D2)) / norm(D)) < 1e-4


        X = reconstruct_pointset(S, 2)

        # Verify that reconstructed `X` is correct up to rotation and translation
        A = [X' ones(size(D, 1))]
        P2 = (A * (A \ P'))'
        @test norm(P - P2) / norm(P) < 1e-3

        X2 = reconstruct_pointset(D, 2)
        X2 = apply_procrustes(X2, X)
        @test norm(X - X2) / norm(X) < 2e-4




        P = randn(2, 400)
        D = pairwise(SqEuclidean(), P, dims = 2)
        W = rand(size(D)...) .> 0.3 # Create a random mask
        W = (W + W') .> 0           # It makes sense for the mask to be symmetric
        W[diagind(W)] .= true
        D0 = W .* D                 # Remove missing entries

        D3, E = rankcomplete_distmat(D0, W, 2, verbose=false)

        @test (norm(D - D3) / norm(D)) < 0.2
        @test (norm(W .* (D - D3)) / norm(D)) < 1e-4

    end


    @testset "procrustes" begin
        @info "Testing procrustes"

        θ = 1
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        t = randn(2)
        Y = randn(2, 40)
        X = R * Y .+ t
        Ri = R'
        ti = -R't

        R2, t2 = procrustes(X, Y)

        @test t2 ≈ ti
        @test R2 ≈ Ri
        @test R2 * X .+ t2 ≈ Y

    end



    @testset "posterior" begin
        @info "Testing posterior"

        N = 10
        σL = 0.1
        σD = 0.01

        P = randn(2, N)
        Pn = P + σL * randn(size(P))
        D = pairwise(Euclidean(), P, dims = 2)
        Dn = D + σD * randn(size(D))
        Dn[diagind(Dn)] .= 0


        distances = []
        noisy_distances = []
        p = 0.5
        for i = 1:N
            for j = i+1:N
                if rand() < p
                    push!(distances, (i, j, D[i, j]))
                    push!(noisy_distances, (i, j, Dn[i, j]))
                end
            end
        end
        @show length(distances)
        @show expected = p * ((N^2 - N) ÷ 2)


        part, chain = posterior(
            Pn,
            noisy_distances;
            nsamples = 2000,
            sampler = NUTS(),
            σL = σL,
            σD = σD,
        )

        @test norm(pmean.(part.P) - P) < norm(Pn - P)

        if isinteractive()
            scatter(part.P[1, :], part.P[2, :], markersize = 6, layout = 2, sp = 1)
            scatter!(P[1, :], P[2, :], lab = "True positions", sp = 1)
            scatter!(Pn[1, :], Pn[2, :], lab = "Measured positions", sp = 1)
            bar!(vec(D), sp = 2, lab = "True dist")
            bar!(vec(Dn), sp = 2, lab = "Measured dist", alpha = 0.7)
            scatter!(vec(part.d), sp = 2, seriestype = :scatter, lab = "") |> display
        end



        part, res =
            posterior(Pn, noisy_distances, LBFGS(), sampler = MAP(), σL = σL, σD = σD)
        @test norm(pmean.(part.P) - P) < norm(Pn - P)

        if isinteractive()
            scatter(part.P[1, :], part.P[2, :], markersize = 6, layout = 2, sp = 1)
            scatter!(P[1, :], P[2, :], lab = "True positions", sp = 1)
            scatter!(Pn[1, :], Pn[2, :], lab = "Measured positions", sp = 1)
            bar!(vec(D), sp = 2, lab = "True dist")
            bar!(vec(Dn), sp = 2, lab = "Measured dist", alpha = 0.7)
            scatter!(vec(part.d), sp = 2, seriestype = :scatter, lab = "") |> display
        end



        aligned_P = align_to_mean(part.P)
        @test tr(pcov(vec(part.P))) > tr(pcov(vec(aligned_P)))



        # Using TDOA measurements instead


        N = 10
        σL = 0.1
        σD = 0.01

        P = 3randn(2, N)
        source = randn(2)
        Pn = P + σL * randn(size(P))
        distances = []
        noisy_distances = []
        p = 0.5
        for i = 1:N
            for j = i+1:N
                if rand() < p
                    di = norm(P[:, i] - source) # Distance from source to i
                    dj = norm(P[:, j] - source) # Distance from source to j
                    tdoa = di - dj # This is the predicted TDOA given the posterior locations
                    push!(distances, (i, j, tdoa))
                    push!(noisy_distances, (i, j, tdoa + σD * randn()))
                end
            end
        end
        @show length(distances)
        @show expected = p * ((N^2 - N) ÷ 2)


        part, chain = posterior(
            [Pn source],
            noisy_distances;
            nsamples = 2000,
            sampler = NUTS(),
            σL = σL,
            σD = σD,
            tdoa = true,
        )

        @test norm(pmean.(part.P[:, 1:end-1]) - P) < norm(Pn - P)

        if isinteractive()
            scatter(part.P[1, 1:end-1], part.P[2, 1:end-1], markersize = 6)
            scatter!(P[1, :], P[2, :], lab = "True positions")
            scatter!(Pn[1, :], Pn[2, :], lab = "Measured positions")
            scatter!(
                [part.P[1, end]],
                [part.P[2, end]],
                m = (:x, 8),
                lab = "Est. Source",
            )
            scatter!([source[1]], [source[2]], m = (:x, 8), lab = "True Source") |>
            display
        end

    end

end



# using SparseArrays, Distances
# using LowRankModels
#
# P = randn(2, 4000)
# D = pairwise(SqEuclidean(), P, dims = 2)
# W = rand(size(D)...) .> 0.98 # Create a random mask
# W = (W + W') .> 0           # It makes sense for the mask to be symmetric
# W[diagind(W)] .= true
# D0 = W .* D                 # Remove missing entries
#
# D0s = sparse(D0) + spdiagm(0=>fill(eps(),size(D0,1)))
#
#
#
# glrm = pca(D0s, 4, offset=true, scale=true)
# # init_svd!(glrm)
# init_nndsvd!(glrm)
# # init_kmeanspp!(glrm)
#
# X,Y,ch = fit!(glrm, SparseProxGradParams(max_iter=400))
# plot(ch.objective, yscale=:log10)
#
# D2 = X'Y
#
# (norm(D - D2) / norm(D))
# @test (norm(D - D2) / norm(D)) < 0.2
# @test (norm(W .* (D - D2)) / norm(D)) < 1e-5
#
#
# D3,E = rankcomplete_distmat(D0, W, 2, tol=1e-4)
# (norm(D - D3) / norm(D))
#
#
# D4,E = D3
# (norm(D - D4) / norm(D))
