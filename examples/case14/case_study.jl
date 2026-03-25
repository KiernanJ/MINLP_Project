"""
Case Study: IEEE 14-Bus Convex AC OPF + Unit Commitment

Builds a repertoire of training data points by sampling linearization voltages u = (node_vr, node_vi)
and solving the convex AC OPF+UC for each sample.
"""

include(joinpath(@__DIR__, "../../src/surrogates.jl"))
using .surrogates
using Random

const FILE_PATH  = joinpath(@__DIR__, "../../data/case14.m")
const SAVE_PATH  = joinpath(@__DIR__, "labeled_data/repertoire.jld2")
const BUS_IDS    = string.(1:14)   # buses "1" through "14"

# Voltage magnitude bounds (from case14.m: Vmin=0.94, Vmax=1.06 for all buses)
const V_MIN = 0.94
const V_MAX = 1.06

# Angle sampling range (radians).
# Nominal case14 angles span [0°, -16°]; we widen slightly to [-25°, 5°] for coverage.
const VA_MIN = -25.0 * π / 180.0
const VA_MAX =   5.0 * π / 180.0


"""
    sample_u(n_samples; rng) -> Vector{Tuple{Dict,Dict}}

Sample `n_samples` linearization points u = (node_vr, node_vi) for the 14-bus system.

Each sample draws voltage magnitudes uniformly from [V_MIN, V_MAX] and angles uniformly
from [VA_MIN, VA_MAX], then converts to rectangular coordinates:
    vr_i = vm_i * cos(va_i)
    vi_i = vm_i * sin(va_i)

Bus 1 (slack/reference) is always fixed to va = 0, so vi["1"] = 0 throughout.
"""
function sample_u(n_samples::Int; rng=Random.GLOBAL_RNG)::Vector{Tuple{Dict{String,Float64}, Dict{String,Float64}}}
    samples = Vector{Tuple{Dict{String,Float64}, Dict{String,Float64}}}(undef, n_samples)

    for k in 1:n_samples
        node_vr = Dict{String,Float64}()
        node_vi = Dict{String,Float64}()

        for i in BUS_IDS
            vm = V_MIN + (V_MAX - V_MIN) * rand(rng)

            if i == "1"   # reference bus: angle locked at 0
                node_vr[i] = vm
                node_vi[i] = 0.0
            else
                va = VA_MIN + (VA_MAX - VA_MIN) * rand(rng)
                node_vr[i] = vm * cos(va)
                node_vi[i] = vm * sin(va)
            end
        end

        samples[k] = (node_vr, node_vi)
    end

    return samples
end


"""
    build_repertoire(n_samples; seed, save) -> Vector{TrainingDataPoint}

Sample `n_samples` linearization points, solve the convex AC OPF+UC for each,
and return (and optionally save) the resulting `TrainingDataPoint` vector.

# Keyword arguments
- `seed::Int = 42`: RNG seed for reproducibility
- `save::Bool = true`: whether to write results to `SAVE_PATH`
"""
function build_repertoire(n_samples::Int; seed::Int=42, save::Bool=true)::Vector{TrainingDataPoint}
    rng    = MersenneTwister(seed)
    params = sample_u(n_samples; rng=rng)

    points = Vector{TrainingDataPoint}(undef, n_samples)
    for (k, (node_vr, node_vi)) in enumerate(params)
        println("Solving instance $k / $n_samples ...")
        points[k] = gather_training_data(FILE_PATH, node_vr, node_vi)
    end

    n_ok = count(p -> p.status in ("OPTIMAL", "LOCALLY_SOLVED"), points)
    println("\nDone: $n_ok / $n_samples instances solved successfully.")

    if save
        save_repertoire(collect(points), SAVE_PATH)
    end

    return collect(points)
end
