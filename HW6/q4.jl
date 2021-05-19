using LinearAlgebra

n = parse(Int, strip(readline()))
c = parse(Int, strip(readline()))

u = parse.(Float64, split(strip(readline())))
A = [parse.(Float64, split(readline())) for i = 1:n]
A = Matrix(hcat(A...)')
##
x = A \ u
r = sqrt(c / (x' * A * x))
x .*= r
for i = 1:n
    print(round(x[i],digits=6), " ")
end
