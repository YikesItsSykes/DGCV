"""
DGCV: Differential Geometry with Complex Variables

This module provides various combinatorial functions used throughout the DGCV package, 
primarily focusing on efficient computation of Cartesian products, permutations, and 
related operations. These functions are optimized for performance and designed to 
handle large inputs by utilizing generators where possible.

Key Functions
-------------
- carProd: Computes the Cartesian product of multiple lists.
- carProdWithOrder: Computes the Cartesian product while removing equivalent permutations.
- carProdWithoutRepl: Computes the Cartesian product, excluding repeated elements.
- chooseOp: Generates combinations with optional order, replacement, and homogeneity degree filtering.
- permSign: Calculates the signature of a permutation.
- permuteTupleEntries: Applies a permutation to the entries of a tuple or list.
- permuteTuple: Applies a permutation to reorder a tuple or list.
- permuteArray: Applies a permutation to the indices of a k-dimensional array.
- alternatingPartOfArray: Computes the alternating part of a multilinear operator represented by an array.

Dependencies
------------
- sympy: For arrays (MutableDenseNDimArray) and mathematical operations.

Notes
-----
This module minimizes dependencies and is critical for handling combinatorics within the DGCV package's computationally intensive operations.

"""




############## dependencies

from sympy import MutableDenseNDimArray


############## general combinatorics

def carProd(*args):
    """
    Compute the Cartesian product of a variable number of lists.
    
    This function takes multiple lists as input and computes their 
    Cartesian product, yielding tuples containing elements from each list.

    Parameters
    ----------
    *args : lists
        The input lists whose Cartesian product is to be computed.

    Returns
    -------
    generator
        A generator yielding tuples representing the Cartesian product.

    Examples
    --------
    >>> list(carProd([1, 2], [3, 4]))
    [(1, 3), (1, 4), (2, 3), (2, 4)]
    
    Notes
    -----
    This function avoids loading the entire Cartesian product into memory 
    at once by using generators.

    Raises
    ------
    TypeError
        If any of the input arguments are not iterable.
    """
    def carProdTwo(arg1, arg2):
        return (j + (k,) for j in arg1 for k in arg2)

    if len(args) == 1:
        return ((j,) for j in args[0])
    else:
        resultLoc = ((j,) for j in args[0])
        for j in range(1, len(args)):
            resultLoc = carProdTwo(resultLoc, args[j])
        return resultLoc

def carProd_with_weights_without_R(*args):
    """
    Form cartesian product (filtered for replacement) of a variable number of lists whose elements are marked with a weight (specifically, list entries should be length 2 lists whose first element goes into the car. prod. space and second element a scalar vaule). Weights are multiplied when elements are joined into a list (i.e., element of the cartesian product space).

    Args:
        args: List
    
    Returns: List
        list of lists marked with weights. Specificially, a list of length 2 lists, each conaintaining a scalar (e.g., number of sympy.Expr) in the second position representing a weight and a list representing the car. prod. element
    
    Raises:
    """
    def prodOfTwo(arg1,arg2):
        return [[j[0]+(k[0],),j[1]*k[1]] for j in arg1 for k in arg2 if len(set(j[0]+(k[0],)))==len(j[0]+(k[0],))]
    if len(args)==1:
        return [[(j[0],), j[1]] for j in args[0]]
    else:
        resultLoc=([(j[0],),j[1]] for j in args[0])
        for j in range(len(args)-1):
            resultLoc=prodOfTwo(resultLoc,list(args[j+1]))
        return resultLoc

def carProdWithOrder(*args):
    """
    Compute the Cartesian product of lists, excluding permutations.
    
    This function computes the Cartesian product of multiple lists and 
    removes elements that are equivalent up to permutation. The input 
    lists are pre-sorted to optimize efficiency, and the function yields 
    unique combinations lazily.

    Parameters
    ----------
    *args : lists
        The input lists whose Cartesian product is to be computed.

    Returns
    -------
    generator
        A generator yielding unique tuples representing the Cartesian 
        product, with permutations removed.

    Examples
    --------
    >>> list(carProdWithOrder([1, 2], [2, 3]))
    [(1, 2), (1, 3), (2, 3)]
    
    Notes
    -----
    The input lists are sorted to ensure permutations are excluded more 
    efficiently. The function yields results lazily for improved memory 
    efficiency.

    Raises
    ------
    TypeError
        If any of the input arguments are not iterable.
    """
    sorted_args = [sorted(arg) for arg in args]
    seen = set()
    
    for combo in carProd(*sorted_args):
        sorted_combo = tuple(combo)
        if sorted_combo not in seen:
            seen.add(sorted_combo)
            yield sorted_combo

def carProdWithoutRepl(*args):
    """
    Compute Cartesian product excluding repeated elements.
    
    This function computes the Cartesian product of multiple lists and 
    filters out tuples that contain repeated values. The function yields 
    tuples lazily for improved memory efficiency.

    Parameters
    ----------
    *args : lists
        The input lists whose Cartesian product is to be computed.

    Returns
    -------
    generator
        A generator yielding tuples that do not contain repeated values.

    Examples
    --------
    >>> list(carProdWithoutRepl([1, 2], [2, 3]))
    [(1, 2), (2, 3)]
    
    Notes
    -----
    This function is memory efficient and excludes tuples with repeated 
    elements in an on-the-fly manner.

    Raises
    ------
    TypeError
        If any of the input arguments are not iterable.
    """
    return (j for j in carProd(*args) if len(set(j)) == len(j))

def carProdWithOrderWithoutRepl(*args):
    """
    Compute Cartesian product excluding permutations and repeated elements.
    
    This function computes the Cartesian product of multiple lists, filters 
    out tuples that contain repeated values, and removes elements equivalent 
    up to permutation by sorting the input lists upfront. The function yields 
    unique tuples lazily for improved memory efficiency.

    Parameters
    ----------
    *args : lists
        The input lists whose Cartesian product is to be computed.

    Returns
    -------
    generator
        A generator yielding unique tuples that do not contain repeated 
        elements or permutations.

    Examples
    --------
    >>> list(carProdWithOrderWithoutRepl([1, 2], [2, 3]))
    [(1, 2), (1, 3)]
    
    Notes
    -----
    By sorting the input lists beforehand and applying filters during the 
    Cartesian product computation, this function minimizes the need to 
    process permutations and repetitions separately.

    Raises
    ------
    TypeError
        If any of the input arguments are not iterable.
    """
    sorted_args = [sorted(arg) for arg in args]
    seen = set()
    
    for combo in carProd(*sorted_args):
        if len(set(combo)) == len(combo):  # Filter out tuples with repeated values
            sorted_combo = tuple(combo)  # Input is already sorted, no need to re-sort here
            if sorted_combo not in seen:
                seen.add(sorted_combo)
                yield sorted_combo

def chooseOp(arg1, arg2, withOrder=False, withoutReplacement=False, restrictHomogeneity=None):
    """
    Generate all possible combinations of length *arg2* containing elements from *arg1*.
    
    The function can apply several filters: excluding permutations, preventing duplicate elements, 
    and restricting combinations to those with a specified homogeneity degree (sum of elements).
    
    Parameters
    ----------
    arg1 : list
        The list of elements from which combinations are drawn.
    arg2 : int
        The length of the combinations to be generated.
    withOrder : bool, optional
        If True, removes equivalent combinations that are permutations of each other.
    withoutReplacement : bool, optional
        If True, prevents duplicate elements within a combination.
    restrictHomogeneity : int, optional
        If set, filters combinations to only include those whose elements sum to the given value.

    Returns
    -------
    generator
        A generator yielding tuples of the specified combinations.
    
    Examples
    --------
    >>> list(chooseOp([1, 2], 2, withOrder=True))
    [(1, 2)]
    
    >>> list(chooseOp([1, 2, 3], 2, restrictHomogeneity=4))
    [(1, 3), (3, 1), (2, 2)]
    
    Notes
    -----
    - The `withOrder` and `withoutReplacement` options control whether permutations 
      and duplicate elements are included.
    - If `restrictHomogeneity` is set, only tuples whose elements sum to the 
      specified value will be returned.

    Raises
    ------
    TypeError
        If the arguments are not in the correct format.
    """
    
    arg1 = [list(arg1)]
    
    # Determine which Cartesian product function to use
    if withOrder:
        if withoutReplacement:
            resultLoc = carProdWithOrderWithoutRepl(*arg2 * arg1)
        else:
            resultLoc = carProdWithOrder(*arg2 * arg1)
    else:
        if withoutReplacement:
            resultLoc = carProdWithoutRepl(*arg2 * arg1)
        else:
            resultLoc = carProd(*arg2 * arg1)
    
    # Apply homogeneity filter if needed
    if isinstance(restrictHomogeneity, int):
        return (j for j in resultLoc if sum(j) == restrictHomogeneity)
    else:
        return resultLoc

def permSign(arg1, returnSorted=False, **kwargs):
    """
    Compute the signature of a permutation of list of integers, and sort it.
    
    The signature (or sign) of a permutation is 1 if the permutation is even, 
    and -1 if the permutation is odd.

    Parameters
    ----------
    arg1 : list
        A list containing a permutation of consecutive integers from 1 to k 

    Returns
    -------
    int
        The signature of the permutation, either 1 (even permutation) or -1 (odd permutation).

    Examples
    --------
    >>> permSign([2, 1, 3])
    -1

    >>> permSign([2, 0, 1])
    1

    Raises
    ------
    ValueError
        If the input is not a valid permutation of consecutive integers.
    """

    # Helper function to count inversions using merge sort
    def merge_count_split_inv(arr):
        if len(arr) < 2:
            return arr, 0
        mid = len(arr) // 2
        left, left_inv = merge_count_split_inv(arr[:mid])
        right, right_inv = merge_count_split_inv(arr[mid:])
        merged, split_inv = merge_and_count(left, right)
        return merged, left_inv + right_inv + split_inv

    def merge_and_count(left, right):
        merged = []
        i = j = 0
        inversions = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inversions += len(left) - i  # All remaining elements in left are inversions
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions

    # Count inversions in the permutation and get the sorted list
    sorted_list, inversions = merge_count_split_inv(arg1)

    # Compute the sign based on the number of inversions
    sign = 1 if inversions % 2 == 0 else -1

    # Return based on the returnSorted flag
    if returnSorted:
        return sign, sorted_list
    else:
        return sign

# def permSignOld(arg1,startAtZero=False):
#     """
#     Computes the signature of a permutation of consecutive integers from 1 up to some integer k

#     Args:
#         arg1: list containing a permutation of consecutive integers from 1 up to some integer k.
#         key word arguments: optional argument *startAtZero=True* makes this function apply to permutations of consecutive integers starting from 0.
    
#     Returns:
#         1 or -1
    
#     Raises:
#     """
#     if startAtZero==True:
#         arg1=[j+1 for j in arg1]
#     powLoc=0
#     for j in range(1,len(arg1)+1):
#         powLoc=powLoc+[k for k in range(len(arg1)) if arg1[k]==j][0]
#         arg1=[k for k in arg1 if k!=j]
#     return int((-1)**powLoc)

############## for tensor caculus
def permuteTupleEntries(arg1, arg2):
    """
    Apply a permutation to the entries of a tuple or list.
    
    This function takes a tuple or list *arg1*, containing integers in the range 
    [0, ..., k-1] for some integer k, and applies a permutation *arg2* to the 
    entries of *arg1*. The result is returned as a new tuple.

    Parameters
    ----------
    arg1 : tuple or list
        A tuple or list containing integers in the range [0, ..., k-1].
    arg2 : list
        A list representing a permutation of [0, 1, ..., k-1].

    Returns
    -------
    tuple
        A new tuple with the entries of *arg1* permuted according to *arg2*.

    Examples
    --------
    >>> permuteTupleEntries((1, 2, 0), [2, 0, 1])
    (0, 1, 2)
    
    Notes
    -----
    - The elements of *arg1* must be valid indices in the permutation *arg2*.
    - If *arg1* contains out-of-range values, the function will raise an error.

    Raises
    ------
    ValueError
        If the values in *arg1* are out of the range defined by the length of *arg2*.
    """
    if not all(0 <= x < len(arg2) for x in arg1):
        raise ValueError("Entries of arg1 must be in the range of [0, ..., len(arg2)-1].")
    
    return tuple(arg2[arg1[j]] for j in range(len(arg1)))

def permuteTuple(arg1, arg2):
    """
    Apply a permutation to the order of a tuple or list.
    
    This function takes a tuple or list *arg1* of length *k*, and applies the 
    permutation *arg2* (a permutation of [0, 1, ..., k-1]) to reorder its 
    elements. The result is returned as a new tuple.

    Parameters
    ----------
    arg1 : tuple or list
        A tuple or list of length *k* whose elements will be permuted.
    arg2 : list
        A list representing a permutation of [0, 1, ..., k-1].

    Returns
    -------
    tuple
        A new tuple with the elements of *arg1* rearranged according to *arg2*.

    Examples
    --------
    >>> permuteTuple((1, 2, 3), [2, 0, 1])
    (3, 1, 2)
    
    Notes
    -----
    - The length of *arg2* must match the length of *arg1*.
    - If *arg2* contains invalid indices, the function will raise an error.

    Raises
    ------
    ValueError
        If the length of *arg2* does not match the length of *arg1*, or if *arg2* 
        contains invalid indices.
    """
    if len(arg1) != len(arg2):
        raise ValueError("The length of arg2 must match the length of arg1.")
    if not all(0 <= x < len(arg1) for x in arg2):
        raise ValueError("arg2 must contain valid indices for arg1.")
    
    return tuple(arg1[j] for j in arg2)

def permuteArray(arg1, arg2):
    """
    Permute the indices of a k-dimensional array representing a multilinear operator.
    
    This function takes a k-dimensional array *arg1* that represents a multilinear operator 
    on n-dimensional space, and applies the permutation *arg2* (a permutation of the 
    coordinate indices [0, 1, ..., k-1]) to the index coordinate tuples of the array. The 
    result is a new array with permuted indices.

    Parameters
    ----------
    arg1 : array-like
        A k-dimensional array representing a multilinear operator.
    arg2 : list
        A permutation of the coordinate indices [0, 1, ..., k-1].

    Returns
    -------
    MutableDenseNDimArray
        A new k-dimensional array with permuted indices.

    Examples
    --------
    >>> from sympy import MutableDenseNDimArray
    >>> A = MutableDenseNDimArray.zeros(2, 2, 2)
    >>> A[0, 1, 0] = 5
    >>> A[1, 0, 1] = 7
    >>> permuteArray(A, [2, 0, 1])
    MutableDenseNDimArray([[[0, 5], [0, 0]], [[0, 0], [7, 0]]])

    Notes
    -----
    - The length of *arg2* must match the number of dimensions of *arg1*.
    - The permutation is applied to the indices of the array, not its values.

    Raises
    ------
    ValueError
        If *arg2* is not a valid permutation of the indices.
    """
    if len(arg2) != len(arg1.shape):
        raise ValueError("The length of arg2 must match the number of dimensions of arg1.")

    # Create a new array to store permuted values
    newArray = MutableDenseNDimArray.zeros(*arg1.shape)

    # Process the generator returned by chooseOp lazily
    for iListLoc in chooseOp(range(arg1.shape[0]), len(arg1.shape)):
        newListLoc = permuteTuple(iListLoc, arg2)
        newArray[newListLoc] = arg1[iListLoc]
    
    return newArray

def alternatingPartOfArray(arg1):
    """
    Calculate the alternating part of a multilinear operator.
    
    This function computes the alternating part of a k-dimensional array 
    representing a multilinear operator. The alternating part is calculated by 
    summing over all possible permutations of the array’s indices, applying 
    the permutation sign (even or odd).

    Parameters
    ----------
    arg1 : Array-like
        A k-dimensional array representing a multilinear operator. The 
        dimensions of the array must be equal (e.g., a square array).

    Returns
    -------
    MutableDenseNDimArray
        The alternating part of the input array.

    Examples
    --------
    >>> from sympy import MutableDenseNDimArray
    >>> A = MutableDenseNDimArray.zeros(3, 3, 3)
    >>> A[0, 1, 2] = 5
    >>> alternatingPartOfArray(A)
    MutableDenseNDimArray([[[0, 0, 0], [0, 0, 5], [0, -5, 0]], [[0, 0, -5], [0, 0, 0], [5, 0, 0]], [[0, 5, 0], [-5, 0, 0], [0, 0, 0]]])

    Notes
    -----
    - The input array must have equal dimensions.
    - The result is obtained by summing over all possible permutations of 
      the indices, weighted by the permutation sign.

    Raises
    ------
    ValueError
        If the input array is not square or has unequal dimensions.
    """
    if isinstance(arg1, MutableDenseNDimArray):
        if len(set(arg1.shape)) == 1:
            permListLoc = chooseOp(range(len(arg1.shape)), len(arg1.shape), withoutReplacement=True, withOrder=True)
            resultArray = MutableDenseNDimArray.zeros(*arg1.shape)
            
            # Process permutations lazily from the generator
            for perm in permListLoc:
                resultArray += permSign(perm) * permuteArray(arg1, perm)
            
            return resultArray
    
    raise ValueError("Input array must have equal dimensions.")



