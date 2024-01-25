import numpy as np
from matplotlib import pyplot as plt
# generate random data
gt_1 = np.array([1,1])
gt_2 = np.array([5,5])
gt_3 = np.array([8,1])
data_1 = np.random.randn(300,2) + gt_1
data_2 = np.random.randn(300,2) + gt_2
data_3 = np.random.randn(300,2) + gt_3
data = np.concatenate((data_1, data_2, data_3),axis = 0)
plt.scatter(data[:,0], data[:,1], s=7,c='b')
plt.show()

import numpy as np


def k_means(data, k, centers):
    m, n = data.shape
    max_iters = 100

    for _ in range(max_iters):
        # Step 2: Assign each observation to the group whose center is the closest
        distances = np.linalg.norm(data[:, np.newaxis, :] - centers, axis=2)
        assignments = np.argmin(distances, axis=1)

        # Step 3: Update cluster centers to be the mean of all points in the group
        new_centers = np.array([data[assignments == j].mean(axis=0) for j in range(k)])

        # Step 4: Check for convergence
        if np.array_equal(centers, new_centers):
            break

        centers = new_centers

    return centers, assignments


# Example usage:
# Assuming data is a 2D array with 2 features
data = np.random.rand(900, 2)
k = 3
initial_centers = np.array([[1, 0], [4, 4], [8, 4]])
final_centers, final_assignments = k_means(data, k, initial_centers)
print("Final Cluster Centers:")
print(final_centers)
print("Final Assignments:")
print(final_assignments)
