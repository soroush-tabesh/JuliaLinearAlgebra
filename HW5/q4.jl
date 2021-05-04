using LinearAlgebra

n = parse(Int, strip(readline()))
A = [parse.(Float64, split(readline())) for i = 1:n]
A = Matrix(hcat(A...)')
b = parse.(Float64, split(strip(readline())))
x̃ = parse.(Float64, split(strip(readline())))
##

function find_min_T(C, x)
    for t = 1:size(C)[2]
        C_t = C[:, 1:t]
        if rank(C_t) < t
            return 0
        end
        if norm(C_t * (C_t \ x) - x) < 1e-3
            return t
        end
    end
end

C_T = hcat([A^i * b for i = 0:n-1]...)
T_min = find_min_T(C_T, x̃)
C_T_min = C_T[:, 1:T_min]
ũ = C_T_min \ x̃
##
function print_matrix(A, dig = 4)
    for i = 1:size(A)[1]
        for j = 1:size(A)[2]
            print(A[i, j], " ")
        end
        println()
    end
end
println(T_min)
print_matrix(round.(C_T_min, digits = 4))
print_matrix(round.(Matrix(ũ'), digits = 4))
