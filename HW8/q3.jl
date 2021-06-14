using LinearAlgebra, Plots
import JSON

function readclassjson(filename)
    datatype_strings = Dict(
        "Int64" => Int64,
        "Float64" => Float64,
        "String" => String,
        "Bool" => Bool,
    )
    raw_data = JSON.parsefile(filename)
    formatted_data = Dict()
    for varname in keys(raw_data)
        vartype = datatype_strings[raw_data[varname]["type"]]
        vardata = raw_data[varname]["data"]
        if isa(vardata, Array)
            vartype = Array{vartype}
        end
        if isa(vardata[1], Array)
            vardata = hcat(vardata...)'
        end
        vardata = convert(vartype, vardata)
        formatted_data[varname] = vardata
    end

    return formatted_data
end

data = readclassjson("./HW8/data/Q3/tomo_data.json")
##
A = collect(data["line_pixel_lengths"]')
y = data["y"]
n = data["npixels"]
##
d = A \ y
X = reshape(d, n, n)
heatmap(
    X,
    yflip = true,
    aspect_ratio = :equal,
    color = :gist_gray,
    cbar = :none,
    framestyle = :none,
)
