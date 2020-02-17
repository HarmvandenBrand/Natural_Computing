import numpy as np
import matplotlib.pyplot as plt

# initialize variables
L = 100
iterations = 1500

# create a random bit sequence
x = np.random.randint(0, 2, L)

# create an empty array that will track the best fitness so far
best_fit = []


for k in range(iterations-1):
    # initialize new bit sequence
    x1 = []
    for i in x:

        # with a probability of 1/L, flip the bit i
        if np.random.randint(0, L) == 23:
            if i:
                x1 = np.append(x1, 0)
            else:
                x1 = np.append(x1, 1)
        else:
            x1 = np.append(x1, i)

    # replace x with x1 if x1 has higher fitness
    if sum(x1) > sum(x):
        x = x1

    # add the best fitness so far to the tracker array
    best_fit = np.append(best_fit, sum(x))
    print(sum(x))


plt.plot(range(iterations-1), best_fit)
plt.title("best fitness")
plt.xlabel("iterations")
plt.ylabel("fitness")
plt.show()
