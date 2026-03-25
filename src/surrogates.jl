module surrogates

using DiffOpt
using Flux
using JuMP
using JLD2

include(joinpath(@__DIR__, "formulation.jl"))
using .formulation

"""
Stores one solved instance of the convex AC OPF+UC problem.

Fields:
- `node_vr`: real voltage parameters used as the linearization point (input u)
- `node_vi`: imaginary voltage parameters used as the linearization point (input u)
- `x_opt`: Dict of optimal variable values keyed by variable name
- `status`: solver termination status string
"""
struct TrainingDataPoint
    node_vr::Dict
    node_vi::Dict
    x_opt::Dict
    status::String
end


"""
    gather_training_data(file_name, node_vr, node_vi)

Solve one instance of the convex AC OPF+UC problem and return a `TrainingDataPoint`.

# Arguments
- `file_name::String`: path to the Matpower `.m` case file
- `node_vr::Dict`: real voltage linearization point for each bus (e.g. `Dict("1" => 1.0, ...)`)
- `node_vi::Dict`: imaginary voltage linearization point for each bus (e.g. `Dict("1" => 0.0, ...)`)

# Returns
A `TrainingDataPoint` with the optimal solution, or an empty `x_opt` if the solve fails.
"""
function gather_training_data(file_name::String, node_vr::Dict, node_vi::Dict)
    model = build_convex_ac_uc(file_name, node_vr, node_vi)
    optimize!(model)

    status = string(termination_status(model))

    if !(status in ("OPTIMAL", "LOCALLY_SOLVED"))
        return TrainingDataPoint(node_vr, node_vi, Dict(), status)
    end

    t = 1  # build_convex_ac_uc uses a single time period

    x_opt = Dict(
        "vr"    => Dict(i => value(model[:vr][i, t])    for i in axes(model[:vr], 1)),
        "vi"    => Dict(i => value(model[:vi][i, t])    for i in axes(model[:vi], 1)),
        "c_ii"  => Dict(i => value(model[:c_ii][i, t])  for i in axes(model[:c_ii], 1)),
        "c_ij"  => Dict(e => value(model[:c_ij][e, t])  for e in axes(model[:c_ij], 1)),
        "s_ij"  => Dict(e => value(model[:s_ij][e, t])  for e in axes(model[:s_ij], 1)),
        "u"     => Dict(g => value(model[:u][g, t])     for g in axes(model[:u], 1)),
        "pg"    => Dict(g => value(model[:pg][g, t])    for g in axes(model[:pg], 1)),
        "qg"    => Dict(g => value(model[:qg][g, t])    for g in axes(model[:qg], 1)),
        "p_fr"  => Dict(b => value(model[:p_fr][b, t])  for b in axes(model[:p_fr], 1)),
        "q_fr"  => Dict(b => value(model[:q_fr][b, t])  for b in axes(model[:q_fr], 1)),
        "p_to"  => Dict(b => value(model[:p_to][b, t])  for b in axes(model[:p_to], 1)),
        "q_to"  => Dict(b => value(model[:q_to][b, t])  for b in axes(model[:q_to], 1)),
        "xi_c"  => Dict(i => value(model[:xi_c][i, t])  for i in axes(model[:xi_c], 1)),
        "xij_c" => Dict(e => value(model[:xij_c][e, t]) for e in axes(model[:xij_c], 1)),
        "xij_s" => Dict(e => value(model[:xij_s][e, t]) for e in axes(model[:xij_s], 1)),
    )

    return TrainingDataPoint(node_vr, node_vi, x_opt, status)
end


# ---------------------------------------------------------------------------
# Repertoire: saving and loading collections of TrainingDataPoints
# ---------------------------------------------------------------------------

"""
    save_repertoire(points, path)

Append `points` (a `Vector{TrainingDataPoint}`) to a JLD2 repertoire file at `path`.
If the file already exists, the new points are merged with the existing ones.
"""
function save_repertoire(points::Vector{TrainingDataPoint}, path::String)
    existing = isfile(path) ? load_repertoire(path) : TrainingDataPoint[]
    merged = vcat(existing, points)
    jldsave(path; repertoire=merged)
    println("Saved $(length(merged)) total points to $path ($(length(points)) new).")
end

"""
    load_repertoire(path)

Load a `Vector{TrainingDataPoint}` from a JLD2 repertoire file at `path`.
"""
function load_repertoire(path::String)::Vector{TrainingDataPoint}
    return load(path, "repertoire")
end


# ---------------------------------------------------------------------------
# Surrogate neural network: encoding and architecture
# ---------------------------------------------------------------------------

"""
    encode_u(point) -> Vector{Float32}

Flatten `(node_vr, node_vi)` from a `TrainingDataPoint` into a 1-D input vector.
Buses are ordered by numeric ID.

Layout: `[vr["1"], …, vr["n"], vi["1"], …, vi["n"]]`
"""
function encode_u(point::TrainingDataPoint)::Vector{Float32}
    bus_ids = sort(collect(keys(point.node_vr)), by=k -> parse(Int, k))
    return Float32[
        [point.node_vr[i] for i in bus_ids]...,
        [point.node_vi[i] for i in bus_ids]...,
    ]
end

"""
    encode_x(point) -> Vector{Float32}

Flatten `x_opt` from a `TrainingDataPoint` into a 1-D target vector.
Keys within each variable group are sorted for a consistent layout across all points.

Layout (groups in order):
`vr, vi, c_ii` (per bus) | `c_ij, s_ij` (per edge) |
`u, pg, qg` (per gen) | `p_fr, q_fr, p_to, q_to` (per branch) |
`xi_c` (per bus) | `xij_c, xij_s` (per edge)
"""
function encode_x(point::TrainingDataPoint)::Vector{Float32}
    x        = point.x_opt
    bus_ids  = sort(collect(keys(x["vr"])),    by=k -> parse(Int, k))
    gen_ids  = sort(collect(keys(x["pg"])),    by=k -> parse(Int, k))
    br_ids   = sort(collect(keys(x["p_fr"])), by=k -> parse(Int, k))
    edge_ids = sort(collect(keys(x["c_ij"])))  # (Int,Int) tuples, lexicographic

    return Float32[
        [x["vr"][i]    for i in bus_ids]...,
        [x["vi"][i]    for i in bus_ids]...,
        [x["c_ii"][i]  for i in bus_ids]...,
        [x["c_ij"][e]  for e in edge_ids]...,
        [x["s_ij"][e]  for e in edge_ids]...,
        [x["u"][g]     for g in gen_ids]...,
        [x["pg"][g]    for g in gen_ids]...,
        [x["qg"][g]    for g in gen_ids]...,
        [x["p_fr"][b]  for b in br_ids]...,
        [x["q_fr"][b]  for b in br_ids]...,
        [x["p_to"][b]  for b in br_ids]...,
        [x["q_to"][b]  for b in br_ids]...,
        [x["xi_c"][i]  for i in bus_ids]...,
        [x["xij_c"][e] for e in edge_ids]...,
        [x["xij_s"][e] for e in edge_ids]...,
    ]
end


"""
    build_ffnn(input_dim, output_dim, hidden_dims; activation) -> Chain

Build a feedforward neural network u → x̂.

# Arguments
- `input_dim::Int`: length of the encoded u vector (output of `encode_u`)
- `output_dim::Int`: length of the encoded x vector (output of `encode_x`)
- `hidden_dims::Vector{Int}`: width of each hidden layer, e.g. `[64, 64]`
- `activation`: activation applied to every hidden layer (default: `relu`)

The output layer is linear (no activation) so the network is unconstrained in range.

# Example
```julia
nn = build_ffnn(28, 231, [128, 128])   # case14: 28 inputs, 231 outputs
û  = nn(encode_u(point))
```
"""
function build_ffnn(
    input_dim::Int,
    output_dim::Int,
    hidden_dims::Vector{Int};
    activation = relu,
)::Chain
    dims = [input_dim; hidden_dims; output_dim]
    layers = []
    for i in 1:(length(dims) - 2)
        push!(layers, Dense(dims[i], dims[i+1], activation))
    end
    push!(layers, Dense(dims[end-1], dims[end]))  # linear output
    return Chain(layers...)
end


# (3) Gradient of the loss function for decision-focused training

end
