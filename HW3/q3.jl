using LinearAlgebra

function mat_pow(A, e)
    eg = eigvecs(A)
    egi = inv(eg)
    d = diagm(eigvals(A) .^ e)
    return real(eg * d * egi)
end
##
t = parse(Int, strip(readline()))
m = parse(Int, strip(readline()))

p = parse.(Float64, split(strip(readline())))
A = Array{Float64}(undef, m, m);
A[1, :] = parse.(Float64, split(strip(readline())))
A[2:end, 1:end-1] = diagm(parse.(Float64, split(strip(readline()))))

##
res = (mat_pow(A, t)) * p
for x in res
    println(round(x))
end
