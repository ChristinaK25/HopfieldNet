from random import randint
import numpy as np
from hopfieldnet.net import HopfieldNetwork  # modified import
from hopfieldnet.trainers import hebbian_training
from matplotlib import pyplot as plt

# Use arrays to perfom name James
j_pattern = np.array([[1, 1, 1, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 0, 1, 0, 0],
                      [1, 1, 1, 0, 0]])

a_pattern = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]])

m_pattern = np.array([[1, 0, 0, 0, 1],
                      [1, 1, 0, 1, 1],
                      [1, 0, 1, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]])

e_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])

s_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1]])

j_pattern *= 2
j_pattern -= 1

a_pattern *= 2
a_pattern -= 1

m_pattern *= 2
m_pattern -= 1

e_pattern *= 2
e_pattern -= 1

s_pattern *= 2
s_pattern -= 1

input_patterns = np.array([j_pattern.flatten(), a_pattern.flatten(), m_pattern.flatten(), e_pattern.flatten(), s_pattern.flatten()])

# first creating the network and then train it with hebbian
network = HopfieldNetwork(35)

hebbian_training(network, input_patterns)

# Create the test patterns by using the training patterns and adding some noise to them
# and use the neural network to denoise them
j_test = j_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    j_test[p] *= -1

j_result = network.run(j_test)

j_result.shape = (7, 5)
j_test.shape = (7, 5)

a_test = a_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    a_test[p] *= -1

a_result = network.run(a_test)

a_result.shape = (7, 5)
a_test.shape = (7, 5)

m_test = m_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    m_test[p] *= -1

m_result = network.run(m_test)

m_result.shape = (7, 5)
m_test.shape = (7, 5)

e_test = e_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    e_test[p] *= -1

e_result = network.run(e_test)

e_result.shape = (7, 5)
e_test.shape = (7, 5)

s_test = s_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    s_test[p] *= -1

s_result = network.run(s_test)

s_result.shape = (7, 5)
s_test.shape = (7, 5)

# Show the results in plots
plt.subplot(3, 4, 1)
plt.imshow(j_test, interpolation="nearest")
plt.subplot(3, 4, 2)
plt.imshow(j_result, interpolation="nearest") #interpolation='nearest' simply displays an image without trying to interpolate
# between pixels if the display resolution is not the same as the image resolution (which is most often the case).
# It will result an image in which pixels are displayed as a square of multiple pixels

plt.subplot(3, 4, 3)
plt.imshow(a_test, interpolation="nearest")
plt.subplot(3, 4, 4)
plt.imshow(a_result, interpolation="nearest")

plt.subplot(3, 4, 5)
plt.imshow(m_test, interpolation="nearest")
plt.subplot(3, 4, 6)
plt.imshow(m_result, interpolation="nearest")

plt.subplot(3, 4, 7)
plt.imshow(e_test, interpolation="nearest")
plt.subplot(3, 4, 8)
plt.imshow(e_result, interpolation="nearest")

plt.subplot(3, 4, 9)
plt.imshow(s_test, interpolation="nearest")
plt.subplot(3, 4, 10)
plt.imshow(s_result, interpolation="nearest")


plt.show()