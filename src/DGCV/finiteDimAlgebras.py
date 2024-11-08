############## dependencies
from sympy import Matrix, MutableDenseNDimArray
import warnings
from .combinatorics import *
from .classesAndVariables import *
from .vectorFieldsAndDifferentialForms import *


############## Algebras




############## linear algebra list processing

def multiply_matrices(A, B):
    """
    Multiplies two matrices A and B, represented as lists of lists.

    Parameters
    ----------
    A : list of lists
        The first matrix (m x n).
    B : list of lists
        The second matrix (n x p).

    Returns
    -------
    list of lists
        The resulting matrix (m x p) after multiplication.

    Raises
    ------
    ValueError
        If the number of columns in A is not equal to the number of rows in B.
    """
    # Get the dimensions of the matrices
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    # Check if matrices are compatible for multiplication
    if cols_A != rows_B:
        raise ValueError("Incompatible matrix dimensions: A is {}x{}, B is {}x{}".format(rows_A, cols_A, rows_B, cols_B))

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):  # or range(rows_B), since cols_A == rows_B
                result[i][j] += A[i][k] * B[k][j]

    return result


def trace_matrix(A):
    """
    Computes the trace of a square matrix A (sum of the diagonal elements).

    Parameters
    ----------
    A : list of lists
        The square matrix.

    Returns
    -------
    trace_value
        The trace of the matrix (sum of the diagonal elements).

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    # Get the dimensions of the matrix
    rows_A, cols_A = len(A), len(A[0])

    # Check if the matrix is square
    if rows_A != cols_A:
        raise ValueError("Trace can only be computed for square matrices. Matrix is {}x{}.".format(rows_A, cols_A))

    # Compute the trace (sum of the diagonal elements)
    trace_value = sum(A[i][i] for i in range(rows_A))

    return trace_value
