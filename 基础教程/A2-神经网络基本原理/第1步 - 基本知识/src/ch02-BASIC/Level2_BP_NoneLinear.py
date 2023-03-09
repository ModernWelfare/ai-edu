# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define function to draw a plot of the given function and input/output points
def draw_fun(X,Y):
    # Define x values to plot the function
    x = np.linspace(1.2,10)
    # Calculate intermediate variables a, b, and c based on x values
    a = x*x
    b = np.log(a)
    c = np.sqrt(b)
    # Plot the function
    plt.plot(x,c)
    # Plot the input/output points
    plt.plot(X,Y,'x')
    # Calculate and plot the derivative of the function
    d = 1/(x*np.sqrt(np.log(x**2)))
    plt.plot(x,d)
    # Show the plot
    plt.show()

# Define forward function that calculates intermediate variables a, b, and c based on x
def forward(x):
    a = x*x
    b = np.log(a)
    c = np.sqrt(b)
    return a,b,c

# Define backward function that calculates the loss and gradients based on intermediate variables and y
def backward(x,a,b,c,y):
    # Calculate the loss between c and y
    loss = c - y
    # Calculate gradients using the chain rule
    delta_c = loss
    delta_b = delta_c * 2 * np.sqrt(b)
    delta_a = delta_b * a
    delta_x = delta_a / 2 / x
    return loss, delta_x, delta_a, delta_b, delta_c

# Define update function that updates x based on the calculated gradient and a lower bound of 1.1
def update(x, delta_x):
    x = x - delta_x
    if x < 1:
        x = 1.1
    return x

# Main function
if __name__ == '__main__':
    # Prompt user to input initial value x and target value y
    print("how to play: 1) input x, 2) calculate c, 3) input target number but not faraway from c")
    print("input x as initial number(1.2,10), you can try 1.3:")
    line = input()
    x = float(line)
    
    a,b,c = forward(x)
    print("c=%f" %c)
    print("input y as target number(0.5,2), you can try 1.8:")
    line = input()
    y = float(line)

    # Set error tolerance for convergence
    error = 1e-3

    # Initialize lists to store x and c values for plotting
    X,Y = [],[]

    # Loop for a maximum of 20 iterations
    for i in range(20):
        # Forward pass to calculate intermediate variables and c
        print("forward...")
        a,b,c = forward(x)
        print("x=%f,a=%f,b=%f,c=%f" %(x,a,b,c))
        # Store x and c values for plotting
        X.append(x)
        Y.append(c)
        # Backward pass to calculate gradients
        print("backward...")
        loss, delta_x, delta_a, delta_b, delta_c = backward(x,a,b,c,y)
        # Check for convergence
        if abs(loss) < error:
            print("done!")
            break
        # Update x based on gradients
        x = update(x, delta_x)
        print("delta_c=%f, delta_b=%f, delta_a=%f, delta_x=%f\n" %(delta_c, delta_b, delta_a, delta_x))

# Draw plot of function and input/output points
draw_fun(X,Y)
