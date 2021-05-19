using LinearAlgebra

A = rand(10000, 500)
b = rand(10000)


##
function kaczmarz_ls(A, b, iter = size(A)[2])
    x = zeros(size(A)[2])
    for k = 1:iter
        i = k % size(A)[1] + 1
        t = ((b[i] - A[i, :] ⋅ x) / (norm(A[i, :])^2)) * A[i, :]
        # println(t[:3])
        x .+= t
    end
    return x
end

kls = kaczmarz_ls(A, b, 100000)
err_kls = norm(b - A * kls)


##
ls = A \ b
err_ls = norm(b - A * ls)
