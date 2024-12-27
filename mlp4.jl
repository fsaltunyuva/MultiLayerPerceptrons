using Flux
using NNlib
using CSV
using DataFrames
using Plots

df = CSV.File("diabetes.csv") |> DataFrame

println(first(df, 5))  # Show the first 5 rows to understand the data

X = select(df, Not(:Outcome)) |> Matrix |> transpose
y = select(df, :Outcome) |> Matrix |> transpose

println(size(X))  # Output will be (num_features, num_samples)

MLP = Chain(Dense(8, 50, gelu), Dropout(0.25),
            Dense(50, 500, gelu), Dropout(0.25),
            Dense(500, 200, gelu), Dropout(0.25),
            Dense(200, 1))
    
train_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true, partial = false);
val_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true)
##Objective functions: weights ---> loss

optim = Flux.setup(Flux.AdamW(0.00001), MLP)  # will store optimiser momentum, etc.

using StatsBase

train_loss = []
validation_loss = []

# Training loop, using the whole data set 1000 times:
using ProgressMeter, ProgressBars, Printf
iter = ProgressBar(1:1_000)

start_time = time()

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

end_time = time()

using BSON
BSON.@save "mlp_model4.bson" MLP

# Load trained model
# BSON.@load "mlp_model.bson" MLP

# Use 50 samples for testing
test_indices = 1:50  # Extract indices for the first 50 samples
new_data = X[:, test_indices]  # Extract 50 samples (shape: 8x10 for 8 features)
true_values = y[:, test_indices]  # Corresponding true target values (shape: 1x10)

# Get predictions for the new data
predictions = MLP(new_data)

# Calculate mean squared error for the 50 test samples
mse_test = Flux.Losses.mse(predictions, true_values)

println("New Data (Input):")
println(new_data)
println("\nTrue Values (Expected):")
println(true_values)
println("\nPredictions:")
println(predictions)
println("\nMean Squared Error for 50 Samples: ", mse_test)
println("Total training time: ", end_time - start_time, " seconds")