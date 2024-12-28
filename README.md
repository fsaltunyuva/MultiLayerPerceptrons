# Multilayer Perceptrons on Julia

Implementing and training 4 different Multilayer Perceptrons by experimenting with different depths, and dropout rates using [diabetes dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). 50 samples were used for the mean squared error.

|   | Optimizer | Layer Structure | Dropout Layers | Activation Function |
| --- | --- | --- | --- | --- |
| Test 1  | Adam (0.00001)  | 8 -> 200 -> 1  | ❌  |  gelu  |
| Test 2  | AdamW (0.00001)  | 8 -> 200 -> 1  | ❌  | gelu  |
| Test 3  | AdamW (0.00001) | 8 –> 50 –> 500 –> 200 –> 1   | ❌  | gelu  |
| Test 4  | AdamW (0.00001) | 8 –> 50 –> 500 –> 200 –> 1   | ✅  | gelu  |

![image](https://github.com/user-attachments/assets/ae17c142-11cf-4035-85b0-004e22dc8902)

### Test 1 (mlp.jl)

**Mean Squared Error for 50 Samples: 0.19419284**

![image](https://github.com/user-attachments/assets/5e875566-5e8a-45be-85f4-063bae2f7785)

### Test 2 (mlp2.jl)

**Mean Squared Error for 50 Samples: 0.17498024**

![image](https://github.com/user-attachments/assets/6b9f7571-c85d-4a65-9489-ed65ea252f26)

### Test 3 (mlp3.jl)

**Mean Squared Error for 50 Samples: 0.14550991**

![image](https://github.com/user-attachments/assets/517d0282-3af6-46e7-b936-89cc28e77af5)

### Test 4 (mlp4.jl)

**Mean Squared Error for 50 Samples: 0.3080534**

![image](https://github.com/user-attachments/assets/08ec805b-c715-4e19-a783-7cc467f33202)
