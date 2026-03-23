include("formulation.jl")
include("utils.jl")



"""
Here is where we will run the case study.


Case study: Unit commitment with AC OPF 

Using convex QC approximation for ACOPF
https://link.springer.com/article/10.1007/s11081-026-10079-4?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20260311&utm_content=10.1007/s11081-026-10079-4
"""

# Initialize the AC OPF with UC

function main()
    convex_ac_uc("data/case14.m", params)
end

function sample_MINLP()
    
end


"""
    solve_instances(parameters::Vector{Vector{Float64}})

Iterates through parameters with a progress bar.
"""
function solve_convex_MINLP(parameters)
    u_dim = length(parameters[1])
    n_cases = length(parameters)
    model, x_vars, u_vars = create_model(u_dim)
    
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
