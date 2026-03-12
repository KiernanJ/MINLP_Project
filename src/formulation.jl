# Optimization Problem Structure
using JuMP, DiffOpt, Ipopt, Gurobi

using PowerModels

# Surrogate Model Architecture
using Flux

"""
Here is where we will maintain the problem structure and the core functions to running this problem.
"""


struct TrainingData
    u_bar::Vector{Float64}
    x_star::Vector{Float64}
end  


function formulate_monolithic(data)
    """
    Function to formulate the AC OPF MIQCQP problem.
    """
    # Parse the file into a Julia Dictionary
    # data = PowerModels.parse_file("case14.m")

    model = Model()

    # --- Variables ---
    # Voltage Magnitude and Angle per bus
    @variable(model, va[i in keys(data["bus"])])
    @variable(model, vm[i in keys(data["bus"])], lower_bound = data["bus"][i]["vmin"], upper_bound = data["bus"][i]["vmax"])

    # Real and Reactive Power per generator
    @variable(model, pg[i in keys(data["gen"])])
    @variable(model, qg[i in keys(data["gen"])])

    # UNIT COMMITMENT: Binary status per generator
    @variable(model, u[i in keys(data["gen"])], Bin)

    # --- Constraints (Example: Power Limits) ---
    for (i, gen) in data["gen"]
        # P_min * u <= Pg <= P_max * u
        @constraint(model, pg[i] >= gen["pmin"] * u[i])
        @constraint(model, pg[i] <= gen["pmax"] * u[i])
    end
end


function solve_monolithic(model::JuMP.Model)
    """
    Function to gather training data for the convex MIQCQP.
        model - JuMP model containing the monolithic MIQCP
    Returns:
        training_data - mapping of the fixed parameters \bar{u} --> x*
    """
    optimize!(model)
    
    u = value.(model[:u])
    x = value.(model[:x])
    
    return TrainingData(u, x)
end