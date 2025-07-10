import numpy as np

def matrixpower(A, m):
    A = np.array(A)

    if A.shape[0] != A.shape[1]:
        return "Matrix must be square"

    if not isinstance(m, int) or m < 1:
        return "Power must be a positive integer"

    res = np.linalg.matrix_power(A, m)
    return res

A = [[1, 2],
     [3, 4]]
m = 3

result = matrixpower(A, m)
print("A^{} is:\n{}".format(m, result))
