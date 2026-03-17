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

    m = Model(Gurobi.Optimizer)


    # --- Variables ---
    # Voltage Magnitude and Angle per bus
    @variable(m, va[i in keys(data["bus"])])
    @variable(m, vm[i in keys(data["bus"])], lower_bound = data["bus"][i]["vmin"], upper_bound = data["bus"][i]["vmax"])

    # Real and Reactive Power per generator
    @variable(m, pg[i in keys(data["gen"])])
    @variable(m, qg[i in keys(data["gen"])])

    # UNIT COMMITMENT: Binary status per generator
    @variable(m, u[i in keys(data["gen"])], Bin)

    # --- Constraints (Example: Power Limits) ---
    for (i, gen) in data["gen"]
        # P_min * u <= Pg <= P_max * u
        @constraint(m, pg[i] >= gen["pmin"] * u[i])
        @constraint(m, pg[i] <= gen["pmax"] * u[i])
    end

    return m
end

function build_single_period_ac_uc(file_path::String)
    # 1. Parse the data
    data = PowerModels.parse_file(file_path)
    
    # Standardize data (adds thermal limits if missing, makes costs uniform)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    # 2. Initialize the JuMP Model
    model = Model()

    # Create dictionaries for easy iteration
    buses = data["bus"]
    gens = data["gen"]
    branches = data["branch"]
    loads = data["load"]

    # 3. Define Variables
    @variable(model, va[keys(buses)]) # Voltage angle
    @variable(model, buses[i]["vmin"] <= vm[i in keys(buses)] <= buses[i]["vmax"]) # Voltage magnitude
    
    @variable(model, u[keys(gens)], Bin) # UNIT COMMITMENT: Binary status
    @variable(model, pg[keys(gens)])     # Active power generation
    @variable(model, qg[keys(gens)])     # Reactive power generation

    # Branch flow variables (from and to ends)
    @variable(model, p_fr[keys(branches)])
    @variable(model, q_fr[keys(branches)])
    @variable(model, p_to[keys(branches)])
    @variable(model, q_to[keys(branches)])

    # 4. Objective Function (Minimize Cost)
    # Note: Using u[i] to handle the fixed no-load cost (c_0) when the unit is ON
    @objective(model, Min, sum(
        gens[i]["cost"][1] * pg[i]^2 + 
        gens[i]["cost"][2] * pg[i] + 
        gens[i]["cost"][3] * u[i] for i in keys(gens)
    ))

    # 5. Constraints
    
    # Reference Bus Angle
    ref_buses = [k for (k,v) in buses if v["bus_type"] == 3]
    for i in ref_buses
        @constraint(model, va[i] == 0.0)
    end

    # Generator Operational Limits tied to Commitment Status
    for (i, gen) in gens
        @constraint(model, pg[i] >= gen["pmin"] * u[i])
        @constraint(model, pg[i] <= gen["pmax"] * u[i])
        @constraint(model, qg[i] >= gen["qmin"] * u[i])
        @constraint(model, qg[i] <= gen["qmax"] * u[i])
    end

    # Branch Power Flow Constraints (AC Polar Formulation)
    for (i, branch) in branches
        f_bus = string(branch["f_bus"])
        t_bus = string(branch["t_bus"])
        
        # Calculate admittance and tap ratios
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        g_fr = branch["g_fr"]; b_fr = branch["b_fr"]
        g_to = branch["g_to"]; b_to = branch["b_to"]
        tm = branch["tap"]
        
        # From-side flows
        @NLconstraint(model, p_fr[i] ==  (g + g_fr)/tm^2 * vm[f_bus]^2 + 
            (-g*tr + b*ti)/tm^2 * (vm[f_bus]*vm[t_bus]*cos(va[f_bus]-va[t_bus])) + 
            (-b*tr - g*ti)/tm^2 * (vm[f_bus]*vm[t_bus]*sin(va[f_bus]-va[t_bus])))
            
        @NLconstraint(model, q_fr[i] == -(b + b_fr)/tm^2 * vm[f_bus]^2 - 
            (-b*tr - g*ti)/tm^2 * (vm[f_bus]*vm[t_bus]*cos(va[f_bus]-va[t_bus])) + 
            (-g*tr + b*ti)/tm^2 * (vm[f_bus]*vm[t_bus]*sin(va[f_bus]-va[t_bus])))

        # To-side flows
        @NLconstraint(model, p_to[i] ==  (g + g_to) * vm[t_bus]^2 + 
            (-g*tr - b*ti)/tm^2 * (vm[t_bus]*vm[f_bus]*cos(va[t_bus]-va[f_bus])) + 
            (-b*tr + g*ti)/tm^2 * (vm[t_bus]*vm[f_bus]*sin(va[t_bus]-va[f_bus])))
            
        @NLconstraint(model, q_to[i] == -(b + b_to) * vm[t_bus]^2 - 
            (-b*tr + g*ti)/tm^2 * (vm[t_bus]*vm[f_bus]*cos(va[t_bus]-va[f_bus])) + 
            (-g*tr - b*ti)/tm^2 * (vm[t_bus]*vm[f_bus]*sin(va[t_bus]-va[f_bus])))
            
        # Thermal Limits
        @constraint(model, p_fr[i]^2 + q_fr[i]^2 <= branch["rate_a"]^2)
        @constraint(model, p_to[i]^2 + q_to[i]^2 <= branch["rate_a"]^2)
    end

    # Nodal Power Balance (Kirchhoff's Current Law)
    for (i, bus) in buses
        # Find all components connected to this bus
        bus_loads = [l for (k,l) in loads if string(l["load_bus"]) == i]
        bus_gens = [k for (k,g) in gens if string(g["gen_bus"]) == i]
        br_fr = [k for (k,b) in branches if string(b["f_bus"]) == i]
        br_to = [k for (k,b) in branches if string(b["t_bus"]) == i]

        pd = sum(l["pd"] for l in bus_loads; init=0.0)
        qd = sum(l["qd"] for l in bus_loads; init=0.0)
        gs = bus["gs"]; bs = bus["bs"]

        @constraint(model, 
            sum(pg[g] for g in bus_gens; init=0.0) - pd - gs * vm[i]^2 == 
            sum(p_fr[b] for b in br_fr; init=0.0) + sum(p_to[b] for b in br_to; init=0.0)
        )
        @constraint(model, 
            sum(qg[g] for g in bus_gens; init=0.0) - qd + bs * vm[i]^2 == 
            sum(q_fr[b] for b in br_fr; init=0.0) + sum(q_to[b] for b in br_to; init=0.0)
        )
    end

    return model, data
end



function build_multi_period_ac_uc(file_path::String, T::Int)
    data = PowerModels.parse_file(file_path)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)

    # Injecting dummy UC parameters since standard .m files lack them
    for (i, gen) in data["gen"]
        gen["ramp_up"] = gen["pmax"] * 0.3    # Can ramp 30% of max capacity per hour
        gen["ramp_down"] = gen["pmax"] * 0.3
        gen["startup_cost"] = 500.0
    end

    model = Model()
    buses = data["bus"]; gens = data["gen"]; branches = data["branch"]; loads = data["load"]

    # --- Time-Indexed Variables ---
    @variable(model, va[keys(buses), 1:T]) 
    @variable(model, buses[i]["vmin"] <= vm[i in keys(buses), t in 1:T] <= buses[i]["vmax"])
    
    @variable(model, u[keys(gens), 1:T], Bin) # On/Off status
    @variable(model, v[keys(gens), 1:T], Bin) # Startup status
    @variable(model, w[keys(gens), 1:T], Bin) # Shutdown status
    
    @variable(model, pg[keys(gens), 1:T])
    @variable(model, qg[keys(gens), 1:T])

    @variable(model, p_fr[keys(branches), 1:T])
    @variable(model, q_fr[keys(branches), 1:T])
    @variable(model, p_to[keys(branches), 1:T])
    @variable(model, q_to[keys(branches), 1:T])

    # --- Objective (Includes startup costs) ---
    @objective(model, Min, sum(
        sum(gens[i]["cost"][1] * pg[i, t]^2 + 
            gens[i]["cost"][2] * pg[i, t] + 
            gens[i]["cost"][3] * u[i, t] + 
            gens[i]["startup_cost"] * v[i, t] for i in keys(gens)) 
        for t in 1:T
    ))

    # --- Constraints ---
    for t in 1:T
        # 1. Network Physics (AC flows & Nodal Balance) for each time period
        # (This uses the exact same logic as Part 1, just adding the `t` index)
        # For brevity in this block, assume the same KCL and AC flow equations as above, 
        # but applied to va[i, t], vm[i, t], pg[i, t], etc.
        
        for (i, gen) in gens
            # 2. Generator Limits
            @constraint(model, pg[i, t] >= gen["pmin"] * u[i, t])
            @constraint(model, pg[i, t] <= gen["pmax"] * u[i, t])

            # 3. Inter-temporal UC Logic (Ramping & State tracking)
            if t > 1
                # Logical relationship between state, startup, and shutdown
                @constraint(model, u[i, t] - u[i, t-1] == v[i, t] - w[i, t])
                
                # Ramping Limits
                @constraint(model, pg[i, t] - pg[i, t-1] <= gen["ramp_up"] * u[i, t-1] + gen["pmin"] * v[i, t])
                @constraint(model, pg[i, t-1] - pg[i, t] <= gen["ramp_down"] * u[i, t] + gen["pmin"] * w[i, t])
            else
                # Initial condition (Assuming all generators start OFF at t=0 for simplicity)
                @constraint(model, u[i, t] == v[i, t])
            end
        end
    end

    return model, data
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