using LinearAlgebra

m = parse(Int, strip(readline()))
G1 = collect(hcat([parse.(Float64, split(readline())) for i = 1:m]...)')
G2 = collect(hcat([parse.(Float64, split(readline())) for i = 1:m]...)')
##

function get_single_flat_col(sz, arr, i)
    mat = zeros(sz)
    mat[:, i] = arr
    return vec(mat)
end

function get_inverter_matrix(mat)
    A = []
    b = []
    sz = size(mat)
    for i = 1:sz[2]
        for j = 1:sz[2]
            push!(A, get_single_flat_col(sz, mat[:, i], j))
            if i == j
                push!(b, 1)
            else
                push!(b, 0)
            end
        end
    end
    A = collect(hcat(A...)')
    b = collect(vcat(b...))
    return A, b
end

function dual_inverter(mat1, mat2)
    A1, b1 = get_inverter_matrix(mat1)
    A2, b2 = get_inverter_matrix(mat2)
    A = [A1; A2]
    b = [b1; b2]
    res = A \ b
    res = reshape(res, size(mat1))
    res = collect(res')
    if norm(I - res * mat1) + norm(I - res * mat2) > 1e-6
        return -1
    end
    return res
end
