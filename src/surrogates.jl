module surrogates


using DiffOpt
# Surrogate Model Architecture
using Flux

"""
This will contain the functions for the surrogate modeling and the embedded convex QCQP.
I will use the convex_ac_uc optimization problem from Constante-Flores and Li 2026.

The parameters that I will train for are the initial voltages for now
Here I define functions that apply to:
(1): Gathering the trianing data
(2): Embedding the convex QCQP formulation into the NN
(3): Calculating the gradient of the loss function to be used for training
(4): 
"""


# struct TrainingData
#     u_bar::Vector{Float64}
#     x_star::Vector{Float64}
# end  

# (1) Embedding the QCQP


end