using Plots
using Clustering
using LinearAlgebra
using Statistics

##
function show_scatter(arr, marker = nothing, sz = nothing, rt = false)
    gr(legends = true)
    if isnothing(sz)
        gr()
    else
        gr(size = sz)
    end
    if isnothing(marker)
        pl = scatter(arr[:, 1], arr[:, 2])
        display(pl)
    else
        pl = scatter(arr[:, 1], arr[:, 2], marker_z = marker)
        display(pl)
    end
    if rt
        return pl
    end
end
##
function get_dists(X)
    n = size(X)[2]
    return [norm(X[:, i] - X[:, j]) for i = 1:n, j = 1:n]
end

function get_laplacian(W)
    return diagm(sum(W, dims = 1)[1, :]) - W
end

function spectral_clustering(X, k, alpha = 10)
    L = get_laplacian(exp.(-alpha * get_dists(X)))
    spec = kmeans(collect((eigvecs(L)[:, 1:k])'), k)
    return spec.assignments
end

clus_spec = spectral_clustering(collect(moons'), 2)

show_scatter(moons, clus_spec)
