using Flux
using NNlib
using CSV
using DataFrames
using Plots

df = CSV.File("diabetes.csv") |> DataFrame # Loading the data from the CSV file

println(first(df, 5))  # Showing the first 5 rows to understand the data

# Extracting the features and target values from the Dataframe
X = select(df, Not(:Outcome)) |> Matrix |> transpose
y = select(df, :Outcome) |> Matrix |> transpose

println(size(X))  # Output will be: features, samples

# Defining the MLP model with gelu activation function
MLP = Chain(Dense(8 => 200), gelu,
    Dense(200 => 200), gelu,
    Dense(200 => 1))

# Creating data loaders for training and validation
train_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true, partial=false);
val_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true)

optim = Flux.setup(Flux.AdamW(0.00001), MLP)  # AdamW optimizer with learning rate 0.00001

using ProgressMeter, ProgressBars, Printf
iter = ProgressBar(1:1_000) # Progress bar for 1000 epochs

start_time = time() # Start time for training

train_loss_for_each_epoch = [] # Store training loss for each epoch
validation_loss_for_each_epoch = [] # Store validation loss for each epoch

for epoch in iter
    train_loss_for_current_epoch = [] # Store training loss for each batch
    validation_loss_for_current_epoch = [] # Store validation loss for each batch

    # Training loop for each batch (same as 0-hellodude.jl on LMS)
    for (x, y) in train_data
        loss_train, grads = Flux.withgradient(MLP) do m
            y_hat = m(x)
            Flux.Losses.mse(y_hat, y)
        end
        Flux.update!(optim, MLP, grads[1])
        push!(train_loss_for_current_epoch, loss_train)
    end

    for (x, y) in val_data
        y_hat = MLP(x)
        loss_val = Flux.Losses.mse(y_hat, y)
        push!(validation_loss_for_current_epoch, loss_val)
    end

    # Storing the mean loss for each epoch to plot the loss curves later
    push!(train_loss_for_each_epoch, mean(train_loss_for_current_epoch))
    push!(validation_loss_for_each_epoch, mean(validation_loss_for_current_epoch))

    # Update the progress bar with the current loss values (by using the last element of the arrays)
    set_postfix(iter, Loss=@sprintf("%.4f, %.4f", train_loss_for_each_epoch[end], validation_loss_for_each_epoch[end]))
end

end_time = time() # End time for training

# To save the trained model
# using BSON
# BSON.@save "mlp_model.bson" MLP

# To load the trained model
# BSON.@load "mlp_model.bson" MLP

# Plotting the loss curves
plot(1:1_000, train_loss_for_each_epoch, label="Training Loss", lw=2, color=:blue) # Plot training loss 
plot!(1:1_000, validation_loss_for_each_epoch, label="Validation Loss", lw=2, color=:red) # Plot validation loss
xlabel!("Epochs") # Label x-axis
ylabel!("Loss") # Label y-axis
title!("Training and Validation Loss") # Title of the plot

# Using 50 samples for testing
new_data = X[:, 1:50]  # Extracting 50 samples
true_values = y[:, 1:50]  # Corresponding true values

predictions = MLP(new_data) # Predictions for the 50 samples

mse_test = Flux.Losses.mse(predictions, true_values) # Mean squared error for the 50 samples

println("True Values (Expected):")
println(true_values)
println("")
println("Predictions:")
println(predictions)
println("")
println("Mean Squared Error for 50 Samples: ", mse_test)
println("Total training time: ", end_time - start_time, " seconds")