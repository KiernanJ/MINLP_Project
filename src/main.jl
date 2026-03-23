include("formulation.jl")
include("utils.jl")



"""
Here is where we will run the case study.


Case study: Unit commitment with AC OPF 

Using convex QC approximation for ACOPF
https://link.springer.com/article/10.1007/s11081-026-10079-4?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20260311&utm_content=10.1007/s11081-026-10079-4
"""

# Initialize the AC OPF with UC

# function main()
#     convex_ac_uc("data/case14.m", params)
# end

# function sample_MINLP()
    
# end


"""
    solve_instances(parameters::Vector{Vector{Float64}})

Here we are just trying to solve one instance of the convex ac uc problem given a set of parameters
"""
function solve_convex_MINLP(file_path::String, parameters)
    u_dim = length(parameters[1])

    # build the optimization problme

    model = convex_ac_uc(file_path, parameters["node_vr"], parameters["node_vi"])
    # solve the optimization problme
    optimize!(model)
    
    x_vars = Dict(); u_vars = Dict() # here are the u and x variables from the solved instances
    # u_vars should just be (n_bus)*2
    # x_vars should contain vr, vi, c_ii, c_ij, s_ij, u, pg, qg, p_fr, q_fr, p_to ?
    


    u_vars[]
    
    results = DataFrame(u = Vector{Float64}[], x_opt = Vector{Float64}[], status = String[])

    # Initialize the Progress Meter
    p_bar = Progress(n_cases; dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:cyan)

    for p in parameters
        try
            # Update fixed parameter variables
            for i in 1:u_dim
                fix(u_vars[i], p[i])
            end
            
            optimize!(model)
            
            stat = string(termination_status(model))
            x_val = (stat == "OPTIMAL") ? value.(x_vars) : Float64[]
            
            push!(results, (u = copy(p), x_opt = x_val, status = stat))
            
        catch e
            # Log error and continue so the whole loop doesn't die
            push!(results, (u = copy(p), x_opt = Float64[], status = "ERROR: $(typeof(e))"))
        finally
            # Increment the progress bar regardless of success/failure
            next!(p_bar)
        end
    end
    
    return results
end


