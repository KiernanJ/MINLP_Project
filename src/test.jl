# # Custom ReLU layer

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/custom-relu.jl)

# We demonstrate how DiffOpt can be used to generate a simple neural network
# unit - the ReLU layer. A neural network is created using Flux.jl and
# trained on the MNIST dataset.

# This tutorial uses the following packages

using JuMP
import DiffOpt
import Ipopt
import ChainRulesCore
import Flux
import MLDatasets
import Statistics
import Base.Iterators: repeated
using LinearAlgebra

# ## The ReLU and its derivative

# Define a relu through an optimization problem solved by a quadratic solver.
# Return the solution of the problem.
# TODO: use HiGHS
function matrix_relu(
    y_data::Matrix;
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer),
)
    layer_size, batch_size = size(y_data)
    empty!(model)
    set_silent(model)
    @variable(model, x[1:layer_size, 1:batch_size] >= 0)
    @variable(model, y[1:layer_size, 1:batch_size] in Parameter.(y_data))
    @objective(model, Min, x[:]'x[:] - 2y[:]'x[:])
    optimize!(model)
    return Float32.(value.(x))
end

# Define the reverse differentiation rule, for the function we defined above.
function ChainRulesCore.rrule(
    ::typeof(matrix_relu),
    y_data::Matrix{T},
) where {T}
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    pv = matrix_relu(y_data; model = model)
    function pullback_matrix_relu(dl_dx)
        ## some value from the backpropagation (e.g., loss) is denoted by `l`
        ## so `dl_dy` is the derivative of `l` wrt `y`
        x = model[:x]::Matrix{JuMP.VariableRef} # load decision variable `x` into scope
        y = model[:y]::Matrix{JuMP.VariableRef} # load parameter variable `y` into scope
        ## set sensitivities (dl/dx)
        for i in eachindex(x)
            DiffOpt.set_reverse_variable(model, x[i], dl_dx[i])
        end
        ## compute grad (dx/dy)
        DiffOpt.reverse_differentiate!(model)
        ## return gradient (dl/dy = dl/dx * dx/dy)
        dl_dy = DiffOpt.get_reverse_parameter.(model, y)
        return (ChainRulesCore.NoTangent(), dl_dy)
    end
    return pv, pullback_matrix_relu
end

# For more details about backpropagation, visit [Introduction, ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/dev/).

# ## Define the network

layer_size = 10
m = Flux.Chain(
    Flux.Dense(784, layer_size), # 784 being image linear dimension (28 x 28)
    matrix_relu,
    Flux.Dense(layer_size, 10), # 10 being the number of outcomes (0 to 9)
    Flux.softmax,
)

# ## Prepare data

N = 1000 # batch size
## Preprocessing train data
imgs = MLDatasets.MNIST(; split = :train).features[:, :, 1:N]
labels = MLDatasets.MNIST(; split = :train).targets[1:N]
train_X = float.(reshape(imgs, size(imgs, 1) * size(imgs, 2), N)) # stack images
train_Y = Flux.onehotbatch(labels, 0:9);
## Preprocessing test data
test_imgs = MLDatasets.MNIST(; split = :test).features[:, :, 1:N]
test_labels = MLDatasets.MNIST(; split = :test).targets[1:N];
test_X = float.(reshape(test_imgs, size(test_imgs, 1) * size(test_imgs, 2), N))
test_Y = Flux.onehotbatch(test_labels, 0:9);

# Define input data
# The original data is repeated `epochs` times because `Flux.train!` only
# loops through the data set once

epochs = 2#50 # ~1 minute (i7 8th gen with 16gb RAM)
## epochs = 100 # leads to 77.8% in about 2 minutes
dataset = repeated((train_X, train_Y), epochs);

# ## Network training

# training loss function, Flux optimizer
custom_loss(m, x, y) = Flux.crossentropy(m(x), y)
opt = Flux.setup(Flux.Adam(), m)

# Train to optimize network parameters

@time Flux.train!(custom_loss, m, dataset, opt);

# Although our custom implementation takes time, it is able to reach similar
# accuracy as the usual ReLU function implementation.

# ## Accuracy results

# Average of correct guesses

accuracy(x, y) = Statistics.mean(Flux.onecold(m(x)) .== Flux.onecold(y));

# Training accuracy

accuracy(train_X, train_Y)

# Test accuracy

println(accuracy(test_X, test_Y))

# Note that the accuracy is low due to simplified training.
# It is possible to increase the number of samples `N`,
# the number of epochs `epoch` and the connectivity `inner`.


# ============================================================


# # ChainRules integration demo: Relaxed Unit Commitment

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/chainrules_unit.jl)

# In this example, we will demonstrate the integration of DiffOpt with
# [ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/),
# the library allowing the definition of derivatives for functions
# that can then be used by automatic differentiation systems.

using JuMP
import DiffOpt
import Plots
import LinearAlgebra: ⋅
import HiGHS
import ChainRulesCore

# ## Unit commitment problem

# We will consider a unit commitment problem, finding the cost-minimizing activation
# of generation units in a power network over multiple time periods.
# The considered constraints include:
# - Demand satisfaction of several loads
# - Ramping constraints
# - Generation limits.

# The decisions are:
# - ``u_{it} \in \{0,1\}``: activation of the ``i``-th unit at time ``t``
# - ``p_{it}``: power output of the ``i``-th unit at time ``t``.

# DiffOpt handles convex optimization problems only, we therefore
# relax the domain of the ``u_{it}`` variables to ``\left[0,1\right]``.

# ## Primal UC problem

# ChainRules defines the differentiation of functions.
# The actual function that is differentiated in the context of DiffOpt is the
# solution map taking in input the problem parameters and returning the solution.

function unit_commitment(
    _load1_demand,
    _load2_demand,
    gen_costs,
    noload_costs;
    model = Model(HiGHS.Optimizer),
    silent = false,
)
    MOI.set(model, MOI.Silent(), silent)

    ## Problem data
    units = [1, 2] # Generator identifiers
    load_names = ["Load1", "Load2"] # Load identifiers
    n_periods = 4 # Number of time periods
    Pmin = Dict(1 => fill(0.5, n_periods), 2 => fill(0.5, n_periods)) # Minimum power output (pu)
    Pmax = Dict(1 => fill(3.0, n_periods), 2 => fill(3.0, n_periods)) # Maximum power output (pu)
    RR = Dict(1 => 0.25, 2 => 0.25) # Ramp rates (pu/min)
    P0 = Dict(1 => 0.0, 2 => 0.0) # Initial power output (pu)

    ## Parameters
    @variable(model, load1_demand[1:n_periods] in Parameter.(_load1_demand)) # Load 1 demand (pu)
    @variable(model, load2_demand[1:n_periods] in Parameter.(_load2_demand)) # Load 2 demand (pu)
    D = Dict("Load1" => load1_demand, "Load2" => load2_demand)
    @variable(model, Cp[1:2] in Parameter.(gen_costs)) # Generation costs ($/pu)
    @variable(model, Cnl[1:2] in Parameter.(noload_costs)) # No-load costs ($)

    ## Variables
    ## Note: u represents the activation of generation units.
    ## Would be binary in the typical UC problem, relaxed here to u ∈ [0,1]
    ## for a linear relaxation.
    @variable(model, 0 <= u[g in units, t in 1:n_periods] <= 1) # Commitment
    @variable(model, p[g in units, t in 1:n_periods] >= 0) # Power output (pu)

    ## Constraints

    ## Energy balance
    @constraint(
        model,
        energy_balance_cons[t in 1:n_periods],
        sum(p[g, t] for g in units) == sum(D[l][t] for l in load_names),
    )

    ## Generation limits
    @constraint(
        model,
        [g in units, t in 1:n_periods],
        Pmin[g][t] * u[g, t] <= p[g, t]
    )
    @constraint(
        model,
        [g in units, t in 1:n_periods],
        p[g, t] <= Pmax[g][t] * u[g, t]
    )

    ## Ramp rates
    @constraint(
        model,
        [g in units, t in 2:n_periods],
        p[g, t] - p[g, t-1] <= 60 * RR[g]
    )
    @constraint(model, [g in units], p[g, 1] - P0[g] <= 60 * RR[g])
    @constraint(
        model,
        [g in units, t in 2:n_periods],
        p[g, t-1] - p[g, t] <= 60 * RR[g]
    )
    @constraint(model, [g in units], P0[g] - p[g, 1] <= 60 * RR[g])

    ## Objective
    @objective(
        model,
        Min,
        sum(
            (Cp[g] * p[g, t]) + (Cnl[g] * u[g, t]) for
            g in units, t in 1:n_periods
        ),
    )

    optimize!(model)
    ## asserting finite optimal value
    @assert termination_status(model) == MOI.OPTIMAL
    ## converting to dense matrix
    return JuMP.value.(p.data)
end

m = Model(HiGHS.Optimizer)
@show unit_commitment(
    [1.0, 1.2, 1.4, 1.6],
    [1.0, 1.2, 1.4, 1.6],
    [1000.0, 1500.0],
    [500.0, 1000.0],
    model = m,
    silent = true,
)

# ## Perturbation of a single input parameter

# Let us vary the demand at the second time frame on both loads:

demand_values = 0.05:0.05:3.0
pvalues = map(demand_values) do di
    return unit_commitment(
        [1.0, di, 1.4, 1.6],
        [1.0, di, 1.4, 1.6],
        [1000.0, 1500.0],
        [500.0, 1000.0];
        silent = true,
    )
end
pflat = [getindex.(pvalues, i) for i in eachindex(pvalues[1])];

# The influence of this variation of the demand is piecewise linear on the
# generation at different time frames:

Plots.scatter(demand_values, pflat; xaxis = ("Demand"), yaxis = ("Generation"))
Plots.title!("Different time frames and generators")
Plots.xlims!(0.0, 3.5)

# ## Forward Differentiation

# Forward differentiation rule for the solution map of the unit commitment problem.
# It takes as arguments:
# 1. the perturbations on the input parameters
# 2. the differentiated function
# 3. the primal values of the input parameters,

# and returns a tuple `(primal_output, perturbations)`, the main primal result
# and the perturbation propagated to this result:

function ChainRulesCore.frule(
    (_, Δload1_demand, Δload2_demand, Δgen_costs, Δnoload_costs),
    ::typeof(unit_commitment),
    load1_demand,
    load2_demand,
    gen_costs,
    noload_costs;
    optimizer = HiGHS.Optimizer,
)
    ## creating the UC model with a DiffOpt optimizer wrapper around HiGHS
    model = DiffOpt.diff_model(optimizer)
    ## building and solving the main model
    pv = unit_commitment(
        load1_demand,
        load2_demand,
        gen_costs,
        noload_costs;
        model = model,
    )
    ## Setting perturbations in the parameters
    DiffOpt.set_forward_parameter.(model, model[:load1_demand], Δload1_demand)
    DiffOpt.set_forward_parameter.(model, model[:load2_demand], Δload2_demand)
    DiffOpt.set_forward_parameter.(model, model[:Cp], Δgen_costs)
    DiffOpt.set_forward_parameter.(model, model[:Cnl], Δnoload_costs)
    ## computing the forward differentiation
    DiffOpt.forward_differentiate!(model)
    ## querying the corresponding perturbation of the decision
    Δp = DiffOpt.get_forward_variable.(model, model[:p])
    return (pv, Δp.data)
end

# We can now compute the perturbation of the output powers `Δpv`
# for a perturbation of the first load demand at time 2:

load1_demand = [1.0, 1.0, 1.4, 1.6]
load2_demand = [1.0, 1.0, 1.4, 1.6]
gen_costs = [1000.0, 1500.0]
noload_costs = [500.0, 1000.0];

# all input perturbations are 0
# except first load at time 2
Δload1_demand = 0 * load1_demand
Δload1_demand[2] = 1.0
Δload2_demand = 0 * load2_demand
Δgen_costs = 0 * gen_costs
Δnoload_costs = 0 * noload_costs
(pv, Δpv) = ChainRulesCore.frule(
    (nothing, Δload1_demand, Δload2_demand, Δgen_costs, Δnoload_costs),
    unit_commitment,
    load1_demand,
    load2_demand,
    gen_costs,
    noload_costs,
)

Δpv

# The result matches what we observe in the previous figure:
# the generation of the first generator at the second time frame (third element on the plot).

# # Reverse-mode differentiation of the solution map

# The `rrule` returns the primal and a pullback.
# The pullback takes a seed for the optimal solution `̄p` and returns
# derivatives with respect to each input parameter of the function.

function ChainRulesCore.rrule(
    ::typeof(unit_commitment),
    load1_demand,
    load2_demand,
    gen_costs,
    noload_costs;
    optimizer = HiGHS.Optimizer,
    silent = false,
)
    model = DiffOpt.diff_model(optimizer)
    ## solve the forward UC problem
    pv = unit_commitment(
        load1_demand,
        load2_demand,
        gen_costs,
        noload_costs;
        model = model,
        silent = silent,
    )
    function pullback_unit_commitment(pb)
        ## set sensitivities
        DiffOpt.set_reverse_variable.(model, model[:p], pb)
        ## compute the gradients
        DiffOpt.reverse_differentiate!(model)
        ## retrieve the gradients with respect to the parameters
        dload1_demand =
            DiffOpt.get_reverse_parameter.(model, model[:load1_demand])
        dload2_demand =
            DiffOpt.get_reverse_parameter.(model, model[:load2_demand])
        dgen_costs = DiffOpt.get_reverse_parameter.(model, model[:Cp])
        dnoload_costs = DiffOpt.get_reverse_parameter.(model, model[:Cnl])
        return (dload1_demand, dload2_demand, dgen_costs, dnoload_costs)
    end
    return (pv, pullback_unit_commitment)
end

# We can set a seed of one on the power of the first generator at the second time frame and zero for all other
# parts of the solution:

(pv, pullback_unit_commitment) = ChainRulesCore.rrule(
    unit_commitment,
    load1_demand,
    load2_demand,
    gen_costs,
    noload_costs;
    optimizer = HiGHS.Optimizer,
    silent = true,
)
dpv = 0 * pv
dpv[1, 2] = 1
dargs = pullback_unit_commitment(dpv)
(dload1_demand, dload2_demand, dgen_costs, dnoload_costs) = dargs;

# The sensitivities with respect to the load demands are:
dload1_demand

# and:
dload2_demand

# The sensitivity of the generation is propagated to the sensitivity of both
# loads at the second time frame.

# This example integrating ChainRules was designed with support
# from [Invenia Technical Computing](https://www.invenia.ca/).
