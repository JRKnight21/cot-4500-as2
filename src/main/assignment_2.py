import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# ===============================================================================
# Question 1: Nevilles Method
# find the 2nd degree interpolating value for f(3.7)
# ===============================================================================

def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    size = len(x_points)
    matrix = np.zeros((size, size))
    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    # populate final matrix (this is the iterative version of the recursion explained in class)
    for i in range(1, size):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]
            denominator = x_points[i] - x_points[i-j]
            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication) / denominator
            matrix[i][j] = coefficient
    
    return matrix[size-1][size-1]

# ================================================================================================
# Question 2: Newton’s forward method
# print out the polynomial approximations for degrees 1, 2, 3
# ================================================================================================

def divided_difference_coefficients(x_points, y_points):
    n = len(x_points)
    coef = list(y_points)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_points[i] - x_points[i-j])
    return coef

def poly_approximation(x, x_points, y_points):
    coefficients = divided_difference_coefficients(x_points, y_points)
    approximations = []
    for degree in range(len(coefficients)):
        approximation = np.zeros_like(x)
        for i in range(degree+1):
            approximation += coefficients[i] * np.power(x - x_points[0], i)
        approximations.append(approximation)
    return coefficients[1:]

# ================================================================================================
# Question 3: Newton’s forward method
# Using the results from 3, approximate f(7.3)
# ================================================================================================

def divided_difference_table(x_points, y_points):
    n = len(x_points)
    table = np.zeros((n, n))
    table[:,0] = y_points
    for j in range(1, n):
        for i in range(n-j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x_points[i+j] - x_points[i])
    return table

def get_approximate_result(matrix, x_points, value):
    n = len(x_points)
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    for i in range(1, n):
        reoccuring_x_span *= (value - x_points[i-1])
        reoccuring_px_result += matrix[0][i] * reoccuring_x_span
    return reoccuring_px_result

# ================================================================================================
# Question 4: Hermite polynomial approximation matrix
# by using the divided difference method
# ================================================================================================

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # get left cell entry
            left: float = matrix[i][j-1]
            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]
            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left
            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i-j+1][0]
            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix

def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    # matrix size changes because of "doubling" up info for hermite 
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))
    # populate x values (make sure to fill every TWO rows)
    for x in range(num_of_points):
        matrix[2*x][0] = x_points[x]
        matrix[2*x+1][0] = x_points[x]
    
    # prepopulate y values (make sure to fill every TWO rows)
    for x in range(num_of_points):
        matrix[2*x][1] = y_points[x]
        matrix[2*x+1][1] = y_points[x]
    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    for x in range(num_of_points):
        matrix[2*x+1][2] = slopes[x]
    
    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)

# ================================================================================================
# Question 5: Cubic spline interpolation
# Find Matrix A, Vector b and Vector x
# ================================================================================================

def generate_spline_matrix(x):
    # number of points
    size = len(x)
    # Create matrix
    A = np.zeros((size, size))

    # Boundary conditions
    # Set the first and last diagonal elements to 1
    A[0, 0] = 1
    A[-1, -1] = 1

    # Interior points
    for i in range(1, size-1):
        # Compute the left and right intervals
        left_interval = x[i] - x[i-1]
        right_interval = x[i+1] - x[i]
        A[i, i-1:i+2] = [left_interval, 2*(left_interval + right_interval), right_interval]

    return A

def generate_vector_b(x, y):
    # number of points
    size = len(x)
    # Create vector
    b = np.zeros(size)

    for i in range(1, size-1):
        # Compute the left and right intervals
        left_interval = x[i] - x[i-1]
        right_interval = x[i+1] - x[i]
        # vector b formula
        b[i] = 3 * ((y[i+1] - y[i])/right_interval - (y[i] - y[i-1])/left_interval)

    return b


if __name__ == "__main__":

    # Question 1
    # point setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7
    nevilles_results = nevilles_method(x_points, y_points, approximating_value)
    print(nevilles_results, end='\n\n')

    # Question 2
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    x = np.array(x_points)

    coefficients = poly_approximation(x, x_points, y_points)
    print(coefficients, end='\n\n')
    
    # Question 3
    divided_table = divided_difference_table(x_points, y_points)
    #np.set_printoptions(precision=7, suppress=True, linewidth=100)
    #print("Divided difference table:")
    #print(divided_table)
    
    approximating_x = 7.3
    final_approximation = get_approximate_result(divided_table, x_points, approximating_x)
    print(final_approximation, end='\n\n')

    # Question 4
    hermite_interpolation()
    print(end='\n')

    # Question 5
    x = [2, 5, 8, 10]
    y = [3, 5, 7, 9]

    A_matrix = generate_spline_matrix(x)
    b_vector = generate_vector_b(x, y)

    # Solve for the coefficients
    x_vector = np.linalg.solve(A_matrix, b_vector)

    print(A_matrix, end='\n\n')
    print(b_vector, end='\n\n')
    print(x_vector)