using LinearAlgebra, DelimitedFiles, Random

data = readdlm("HW7/data/data2.txt")
sfl = shuffle(1:size(data)[1])
lbl = Int.(data[sfl, end])
data = data[sfl, 1:end-1]

##
function get_slice(arr, k, i)
    slr = zeros(Bool, (size(arr)[1]))
    slr[(size(arr)[1]÷k*(i-1)+1):(size(arr)[1]÷k*i)] .= 1
    return arr[slr, :], arr[(!).(slr), :]
end

test_data, train_data = get_slice(data, 5, 5)
test_lbl, train_lbl = get_slice(lbl, 5, 5)

##

function perceptron_grouped(X, y, max_iter = 100)
    w = zeros(size(X)[2])
    for t = 1:max_iter
        flag = false
        for i = 1:size(X)[1]
            if y[i] * (w ⋅ X[i, :]) <= 0
                w .+= y[i] .* X[i, :]
                flag = true
                break
            end
        end
        if !flag
            break
        end
    end
    return w
end

function predict_w(X, w)
    res = X * w
    res = (2 * (res .> 0)) .- 1
    return res
end

w = perceptron_grouped(train_data, train_lbl)
test_pred = predict_w(test_data, w)
acc = count((test_pred - test_lbl) .== 0) / length(test_lbl)
println(acc)
