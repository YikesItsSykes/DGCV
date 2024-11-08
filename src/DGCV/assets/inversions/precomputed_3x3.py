import sympy as sp

l1l1, l1l2, l1l3, l2l1, l2l2, l2l3, l3l1, l3l2, l3l3 = sp.symbols('l1l1, l1l2, l1l3, l2l1, l2l2, l2l3, l3l1, l3l2, l3l3')

inverse_formulas = {
    3: {
        'det': l1l1*l2l2*l3l3 - l1l1*l2l3*l3l2 - l1l2*l2l1*l3l3 + l1l2*l2l3*l3l1 + l1l3*l2l1*l3l2 - l1l3*l2l2*l3l1,
        'adjugate': [[l2l2*l3l3 - l2l3*l3l2, -l1l2*l3l3 + l1l3*l3l2, l1l2*l2l3 - l1l3*l2l2], [-l2l1*l3l3 + l2l3*l3l1, l1l1*l3l3 - l1l3*l3l1, -l1l1*l2l3 + l1l3*l2l1], [l2l1*l3l2 - l2l2*l3l1, -l1l1*l3l2 + l1l2*l3l1, l1l1*l2l2 - l1l2*l2l1]],
    }
}
