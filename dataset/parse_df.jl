
using DataFrames
using JLD2
using CSV

file = jldopen("df.jld2")
df = file["df"]

df = df[:, [:move, :result]]
CSV.write("data.csv", df)
