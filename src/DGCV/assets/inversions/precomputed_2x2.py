import sympy as sp

l1l1, l1l2, l2l1, l2l2 = sp.symbols('l1l1, l1l2, l2l1, l2l2')

inverse_formulas = {
    2: {
        'det': l1l1*l2l2 - l1l2*l2l1,
        'adjugate': [[l2l2, -l1l2], [-l2l1, l1l1]],
    }
}
