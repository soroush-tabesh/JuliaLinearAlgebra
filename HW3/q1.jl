using LinearAlgebra

function get_independents(
    A,
    atol = 0,
    rtol = atol > 0 ? 0 : size(A, 1) * eps(eltype(A)),
)
    k = rank(A; atol = atol, rtol = rtol)
    q = qr(A, Val(true))
    return A[:, q.p[1:k]]
end

function fit_ratio(A, b)
    A = get_independents(A, 1e-6)
    err = A * (A \ b) - b
    return norm(err)
end

function rm_row(M, i)
    return [M[1:i-1, :]; M[i+1:end, :]]
end

# ##
# A = [1.00000000001 1 1; 1 1 1; 1 1 1; 1 1 1]
# b = [1; 1; 2; 1]
# Ab = [A b]
#
# res1 = fit_ratio(A, b)
##
n = strip(readline())
n = parse(Int32, n)

A = [parse.(Float64, split(readline())) for i = 1:n]
A = hcat(A...)'

b = parse.(Float64, split(readline()))

##
errs = [fit_ratio(rm_row(A, i), rm_row(b, i)) for i = 1:(size(A)[1])]
push!(errs, fit_ratio(A, b))
errs

if sum(errs) < 1e-3
    println("NO SENSORS HAVE FAILED")
else
    println(argmin(errs))
end
