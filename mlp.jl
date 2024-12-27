using Flux
using NNlib


X = randn(100,1000) ### Here the number of samples is 100, the number of features is 10!!!
y = randn(1, 1000)

MLP = Chain(Dense(100=>200), gelu, 
            Dense(200=>200), gelu, 
            Dense(200=>1))
            
train_data = Flux.DataLoader((X, y), batchsize=64, shuffle=true, partial = false);

for (x,y) in train_data
    println(x |> size, y |> size)
end

val_data = Flux.DataLoader((randn(100,1000), randn(1, 1000)), batchsize=64, shuffle=true);
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
BSON.@load "mlp_model.bson" MLP

new_data = randn(100, 5)  # Example: 5 new samples with 100 features each

predictions = MLP(new_data)
