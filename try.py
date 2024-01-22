import numpy as np

matrix_2d = np.array([[1, 2, 3, 4],
                      [4, 5, 6, 7],
                      [7, 8, 9, 10]])
sum_result = np.sum(matrix_2d, axis=0)

print("2D array:\n", matrix_2d)
print("Sum of rows (resulting in a 1D array with size D):", sum_result)