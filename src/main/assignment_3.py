import numpy as np
#1
def function_given(a, b):
    return (a - b**2)
def eulers_method(a, b, c, iterations):
    d = (c - a) / iterations
    for unused_variable in range(iterations):
        b = b + (d * function_given(a, b))
        a = a + d
    print(b, "\n")
#2
def func(a, b):
    return (a - b**2)
def runge_kutta(a, b, c, iterations):
    d = (c - a) / iterations 
    for another_unused_variable in range(iterations):
        e_1 = d * func(a, b)
        e_2 = d * func((a + (d / 2)), (b + (e_1 / 2)))
        e_3 = d * func((a + (d / 2)), (b + (e_2 / 2)))
        e_4 = d * func((a + d), (b + e_3))
        b = b + (1 / 6) * (e_1 + (2 * e_2) + (2 * e_3) + e_4)
        a = a + d
    print(b, "\n")
#3
def gaussian_elimination(gaussian_matrix):
    size = gaussian_matrix.shape[0]
    for i in range(size):
        pivot = i
        while gaussian_matrix[pivot, i] == 0:
            pivot += 1
        gaussian_matrix[[i, pivot]] = gaussian_matrix[[pivot, i]]
        for j in range(i + 1, size):
            factor = gaussian_matrix[j, i] / gaussian_matrix[i, i]
            gaussian_matrix[j, i:] = gaussian_matrix[j, i:] - factor * gaussian_matrix[i, i:]
    inputs = np.zeros(size)
    for i in range(size - 1, -1, -1):
        inputs[i] = (gaussian_matrix[i, -1] - np.dot(gaussian_matrix[i, i: -1], inputs[i:])) / gaussian_matrix[i, i]  
    final_answer = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])])
    print(final_answer, "\n")
#4
def lu_factorization(lu_matrix):
    size = lu_matrix.shape[0]
    l_factor = np.eye(size)
    u_factor = np.zeros_like(lu_matrix)
    for i in range(size):
        for j in range(i, size):
            u_factor[i, j] = (lu_matrix[i, j] - np.dot(l_factor[i, :i], u_factor[:i, j]))
        for j in range(i + 1, size):
            l_factor[j, i] = (lu_matrix[j, i] - np.dot(l_factor[j, :i], u_factor[:i, i])) / u_factor[i, i]
    determinant = np.linalg.det(lu_matrix)
    print(determinant, "\n")
    print(l_factor, "\n")
    print(u_factor, "\n")
#5
def diagonally_dominant(dd_matrix, n):
    for i in range(0, n):
        total = 0
        for j in range(0, n):
            total = total + abs(dd_matrix[i][j])  
        total = total - abs(dd_matrix[i][i])
    if abs(dd_matrix[i][i]) < total:
        print("False\n")
    else:
        print("True\n")
#6
def positive_definite(pd_matrix):
    eigenvalues = np.linalg.eigvals(pd_matrix)
    if np.all(eigenvalues > 0):
        print("True\n")
    else:
        print("False\n")
if __name__ == "__main__":
    #1
    a_0 = 0
    b_0 = 1
    c = 2
    iterations = 10
    eulers_method(a_0, b_0, c, iterations)
    #2
    a_0 = 0
    b_0 = 1
    c = 2
    iterations = 10
    runge_kutta(a_0, b_0, c, iterations)
    #3
    gaussian_matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
    gaussian_elimination(gaussian_matrix)
    #4
    lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double)
    lu_factorization(lu_matrix)
    #5
    n = 5
    dd_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
    diagonally_dominant(dd_matrix, n)
    #6
    pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    positive_definite(pd_matrix)
