module utils

using JLD2
using DataFrames
using ProgressMeter

"""
    Here is where we can run the utilities for running the main case study.
"""
export save_results



"""
    save_results(df::DataFrame, filename::String)
"""
function save_results(df, filename)
    if isempty(df)
        @warn "DataFrame is empty. Nothing to save."
        return
    end
    jldsave(filename; data=df)
    println("\n✅ Saved $(nrow(df)) instances to $filename")
end

end