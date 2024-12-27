using Flux
using NNlib
using CSV
using DataFrames

df = CSV.File("diabetes.csv") |> DataFrame

println(first(df, 5))  # Show the first 5 rows to understand the data

X = select(df, Not(:Outcome)) |> Matrix |> transpose
y = select(df, :Outcome) |> Matrix |> transpose

println(size(X))  # Output will be (num_features, num_samples)

MLP = Chain(Dense(8=>200), gelu, 
            Dense(200=>200), gelu, 
            Dense(200=>1))
            
train_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true, partial = false);
val_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true)
##Objective functions: weights ---> loss

optim = Flux.setup(Flux.Adam(0.00001), MLP)  # will store optimiser momentum, etc.

using StatsBase

train_loss = []
validation_loss = []

# Training loop, using the whole data set 1000 times:
using ProgressMeter, ProgressBars, Printf
iter = ProgressBar(1:1_000)

for epoch in iter
    for (x, y) in train_data
        ## If you like carry training on GPU,
        ## x,y = x |> gpu, y |>gpu
        loss_train, grads = Flux.withgradient(MLP) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.Losses.mse(y_hat, y)
            
    end
        push!(train_loss, loss_train)
        Flux.update!(optim, MLP, grads[1])
    for (x, y) in val_data
            y_hat = MLP(x)
            loss_val = Flux.Losses.mse(y_hat, y)
            push!(validation_loss, loss_val)   
    end
    set_postfix(iter, Loss=  @sprintf("%.4f, %.4f", mean(loss_train), mean(validation_loss)))

    end
end

using BSON
BSON.@save "mlp_model.bson" MLP

# Load trained model
# BSON.@load "mlp_model.bson" MLP

# Use the first 5 samples for testing
new_data = X[:, 1:5]  # Extract 5 samples (shape will be 10x5 for 10 features)
true_values = y[:, 1:5]  # Corresponding true target values (shape will be 1x5)

# Get predictions from the trained model
predictions = MLP(new_data)

println("New Data (Input):")
println(new_data)
println("\nTrue Values (Expected):")
println(true_values)
println("\nPredictions:")
println(predictions)