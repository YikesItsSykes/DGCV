import sympy as sp

l1l1, l1l2, l1l3, l1l4, l2l1, l2l2, l2l3, l2l4, l3l1, l3l2, l3l3, l3l4, l4l1, l4l2, l4l3, l4l4 = sp.symbols('l1l1, l1l2, l1l3, l1l4, l2l1, l2l2, l2l3, l2l4, l3l1, l3l2, l3l3, l3l4, l4l1, l4l2, l4l3, l4l4')

inverse_formulas = {
    4: {
        'det': l1l1*l2l2*l3l3*l4l4 - l1l1*l2l2*l3l4*l4l3 - l1l1*l2l3*l3l2*l4l4 + l1l1*l2l3*l3l4*l4l2 + l1l1*l2l4*l3l2*l4l3 - l1l1*l2l4*l3l3*l4l2 - l1l2*l2l1*l3l3*l4l4 + l1l2*l2l1*l3l4*l4l3 + l1l2*l2l3*l3l1*l4l4 - l1l2*l2l3*l3l4*l4l1 - l1l2*l2l4*l3l1*l4l3 + l1l2*l2l4*l3l3*l4l1 + l1l3*l2l1*l3l2*l4l4 - l1l3*l2l1*l3l4*l4l2 - l1l3*l2l2*l3l1*l4l4 + l1l3*l2l2*l3l4*l4l1 + l1l3*l2l4*l3l1*l4l2 - l1l3*l2l4*l3l2*l4l1 - l1l4*l2l1*l3l2*l4l3 + l1l4*l2l1*l3l3*l4l2 + l1l4*l2l2*l3l1*l4l3 - l1l4*l2l2*l3l3*l4l1 - l1l4*l2l3*l3l1*l4l2 + l1l4*l2l3*l3l2*l4l1,
        'adjugate': [[l2l2*l3l3*l4l4 - l2l2*l3l4*l4l3 - l2l3*l3l2*l4l4 + l2l3*l3l4*l4l2 + l2l4*l3l2*l4l3 - l2l4*l3l3*l4l2, -l1l2*l3l3*l4l4 + l1l2*l3l4*l4l3 + l1l3*l3l2*l4l4 - l1l3*l3l4*l4l2 - l1l4*l3l2*l4l3 + l1l4*l3l3*l4l2, l1l2*l2l3*l4l4 - l1l2*l2l4*l4l3 - l1l3*l2l2*l4l4 + l1l3*l2l4*l4l2 + l1l4*l2l2*l4l3 - l1l4*l2l3*l4l2, -l1l2*l2l3*l3l4 + l1l2*l2l4*l3l3 + l1l3*l2l2*l3l4 - l1l3*l2l4*l3l2 - l1l4*l2l2*l3l3 + l1l4*l2l3*l3l2], [-l2l1*l3l3*l4l4 + l2l1*l3l4*l4l3 + l2l3*l3l1*l4l4 - l2l3*l3l4*l4l1 - l2l4*l3l1*l4l3 + l2l4*l3l3*l4l1, l1l1*l3l3*l4l4 - l1l1*l3l4*l4l3 - l1l3*l3l1*l4l4 + l1l3*l3l4*l4l1 + l1l4*l3l1*l4l3 - l1l4*l3l3*l4l1, -l1l1*l2l3*l4l4 + l1l1*l2l4*l4l3 + l1l3*l2l1*l4l4 - l1l3*l2l4*l4l1 - l1l4*l2l1*l4l3 + l1l4*l2l3*l4l1, l1l1*l2l3*l3l4 - l1l1*l2l4*l3l3 - l1l3*l2l1*l3l4 + l1l3*l2l4*l3l1 + l1l4*l2l1*l3l3 - l1l4*l2l3*l3l1], [l2l1*l3l2*l4l4 - l2l1*l3l4*l4l2 - l2l2*l3l1*l4l4 + l2l2*l3l4*l4l1 + l2l4*l3l1*l4l2 - l2l4*l3l2*l4l1, -l1l1*l3l2*l4l4 + l1l1*l3l4*l4l2 + l1l2*l3l1*l4l4 - l1l2*l3l4*l4l1 - l1l4*l3l1*l4l2 + l1l4*l3l2*l4l1, l1l1*l2l2*l4l4 - l1l1*l2l4*l4l2 - l1l2*l2l1*l4l4 + l1l2*l2l4*l4l1 + l1l4*l2l1*l4l2 - l1l4*l2l2*l4l1, -l1l1*l2l2*l3l4 + l1l1*l2l4*l3l2 + l1l2*l2l1*l3l4 - l1l2*l2l4*l3l1 - l1l4*l2l1*l3l2 + l1l4*l2l2*l3l1], [-l2l1*l3l2*l4l3 + l2l1*l3l3*l4l2 + l2l2*l3l1*l4l3 - l2l2*l3l3*l4l1 - l2l3*l3l1*l4l2 + l2l3*l3l2*l4l1, l1l1*l3l2*l4l3 - l1l1*l3l3*l4l2 - l1l2*l3l1*l4l3 + l1l2*l3l3*l4l1 + l1l3*l3l1*l4l2 - l1l3*l3l2*l4l1, -l1l1*l2l2*l4l3 + l1l1*l2l3*l4l2 + l1l2*l2l1*l4l3 - l1l2*l2l3*l4l1 - l1l3*l2l1*l4l2 + l1l3*l2l2*l4l1, l1l1*l2l2*l3l3 - l1l1*l2l3*l3l2 - l1l2*l2l1*l3l3 + l1l2*l2l3*l3l1 + l1l3*l2l1*l3l2 - l1l3*l2l2*l3l1]],
    }
}
