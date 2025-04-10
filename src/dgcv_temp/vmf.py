import sympy as sp
from sympy import I

from ._safeguards import get_variable_registry
from .combinatorics import carProd_with_weights_without_R, permSign
from .conversions import _allToSym


def _coeff_dict_formatter(
    varSpace,coeff_dict,valence,total_degree,_varSpace_type,data_shape
):
    """
    Helper function to populate conversion dicts for tensor field classes
    """
    variable_registry = get_variable_registry()
    CVS = variable_registry["complex_variable_systems"]

    exhaust1 = list(varSpace)
    populate = {
        "compCoeffDataDict": dict(),
        "realCoeffDataDict": dict(),
        "holVarDict": dict(),
        "antiholVarDict": dict(),
        "realVarDict": dict(),
        "imVarDict": dict(),
        "preProcessMinDataToHol": dict(),
        "preProcessMinDataToReal": dict(),
    }
    if _varSpace_type == "real":
        for var in varSpace:
            varStr = str(var)
            if var in exhaust1:
                for parent in CVS.values():
                    if varStr in parent["variable_relatives"]:
                        cousin = (
                            set(
                                parent["variable_relatives"][varStr][
                                    "complex_family"
                                ][2:]
                            )
                            - {var}
                        ).pop()
                        if cousin in exhaust1:
                            exhaust1.remove(cousin)
                        if (
                            parent["variable_relatives"][varStr][
                                "complex_positioning"
                            ]
                            == "real"
                        ):
                            realVar = var
                            exhaust1.remove(var)
                            imVar = cousin
                        else:
                            realVar = cousin
                            exhaust1.remove(var)
                            imVar = var
                        holVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][0]
                        antiholVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][1]
                        populate["holVarDict"][holVar] = [realVar, imVar]
                        populate["antiholVarDict"][antiholVar] = [
                            realVar,
                            imVar,
                        ]
                        populate["realVarDict"][realVar] = [holVar, antiholVar]
                        populate["imVarDict"][imVar] = [holVar, antiholVar]
    else:  # _varSpace_type == 'complex'
        for var in varSpace:
            varStr = str(var)
            if var in exhaust1:
                for parent in CVS.values():
                    if varStr in parent["variable_relatives"]:
                        cousin = (
                            set(
                                parent["variable_relatives"][varStr][
                                    "complex_family"
                                ][:2]
                            )
                            - {var}
                        ).pop()
                        if cousin in exhaust1:
                            exhaust1.remove(cousin)
                        if (
                            parent["variable_relatives"][varStr][
                                "complex_positioning"
                            ]
                            == "holomorphic"
                        ):
                            holVar = var
                            exhaust1.remove(var)
                            antiholVar = cousin
                        else:
                            holVar = cousin
                            exhaust1.remove(var)
                            antiholVar = var
                        realVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][2]
                        imVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][3]
                        populate["holVarDict"][holVar] = [realVar, imVar]
                        populate["antiholVarDict"][antiholVar] = [
                            realVar,
                            imVar,
                        ]
                        populate["realVarDict"][realVar] = [holVar, antiholVar]
                        populate["imVarDict"][imVar] = [holVar, antiholVar]
    new_realVarSpace = tuple(populate["realVarDict"].keys())
    new_holVarSpace = tuple(populate["holVarDict"].keys())
    new_antiholVarSpace = tuple(populate["antiholVarDict"].keys())
    new_imVarSpace = tuple(populate["imVarDict"].keys())

    if len(valence) == 0:
        if _varSpace_type == "real":
            populate["realCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["compCoeffDataDict"] = [
                new_holVarSpace + new_antiholVarSpace,
                {(0,) * total_degree: coeff_dict[(0,) * total_degree]},
            ]
        else:
            populate["compCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["realCoeffDataDict"] = [
                new_realVarSpace + new_imVarSpace,
                {(0,) * total_degree: coeff_dict[(0,) * total_degree]},
            ]
    else:

        def _retrieve_indices(term, typeSet=None):
            if typeSet == "symb":
                dictLoc = populate["realVarDict"] | populate["imVarDict"]
                refTuple = new_holVarSpace + new_antiholVarSpace
                termList = dictLoc[term]
            elif typeSet == "real":
                dictLoc = populate["holVarDict"] | populate["antiholVarDict"]
                refTuple = new_realVarSpace + new_imVarSpace
                termList = dictLoc[term]
            index_a = refTuple.index(termList[0])
            index_b = refTuple.index(termList[1], index_a + 1)
            return [index_a, index_b]

        # set up the conversion dicts for index conversion
        if _varSpace_type == "real":
            populate["preProcessMinDataToHol"] = {
                j: _retrieve_indices(varSpace[j], "symb")
                for j in range(len(varSpace))
            }

        else:  # if _varSpace_type == 'complex'
            populate["preProcessMinDataToReal"] = {
                j: _retrieve_indices(varSpace[j], "real")
                for j in range(len(varSpace))
            }

        # coordinate VF and DF conversion
        def decorateWithWeights(index, variance_rule, target="symb"):
            if variance_rule == 0:  # covariant case
                covariance = True
            else:                   # contravariant case
                covariance = False

            if target == "symb":
                if varSpace[index] in variable_registry['conversion_dictionaries']['real_part'].values():
                    holScale = sp.Rational(1, 2) if covariance else 1 # D_z (d_z) coeff of D_x (d_x)
                    antiholScale = sp.Rational(1, 2) if covariance else 1 # D_BARz (d_BARz) coeff of D_x (d_x)
                else:
                    holScale = -I / 2 if covariance else I  # D_z (d_z) coeff of D_y (d_y)
                    antiholScale = I / 2 if covariance else -I  # d_BARz (D_BARz) coeff of d_y (D_y)
                return [
                    [populate["preProcessMinDataToHol"][index][0], holScale],
                    [
                        populate["preProcessMinDataToHol"][index][1],
                        antiholScale,
                    ],
                ]
            else:  # converting from hol to real
                if varSpace[index] in variable_registry['conversion_dictionaries']['holToReal']:
                    realScale = 1 if covariance else sp.Rational(1,2)   # D_x (d_x) coeff in D_z (d_z)
                    imScale = I if covariance else -I*sp.Rational(1,2)  # D_y (d_y) coeff in D_z (d_z)
                else:
                    realScale = 1 if covariance else sp.Rational(1,2)   # D_x (d_x) coeff of D_BARz (d_BARz)
                    imScale = -I if covariance else I*sp.Rational(1,2) # D_y (d_y) coeff of D_BARz (d_BARz)
                return [
                    [populate["preProcessMinDataToReal"][index][0], realScale],
                    [populate["preProcessMinDataToReal"][index][1], imScale],
                ]

        otherDict = dict()
        for term_index, term_coeff in coeff_dict.items():
            if _varSpace_type == "real":
                reformatTarget = "symb"
            else:
                reformatTarget = "real"
            termIndices = [
                decorateWithWeights(k, valence[j], target=reformatTarget) for j,k in enumerate(term_index)
            ]
            prodWithWeights = carProd_with_weights_without_R(*termIndices)
            prodWWRescaled = [[tuple(k[0]), term_coeff * k[1]] for k in prodWithWeights]
            minimal_term_set = _shape_basis(prodWWRescaled,data_shape)
            for term in minimal_term_set:
                if term[0] in otherDict:
                    oldVal = otherDict[term[0]]
                    otherDict[term[0]] = _allToSym(oldVal + term[1])
                else:
                    otherDict[term[0]] = _allToSym(term[1])

        if _varSpace_type == "real":
            populate["realCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["compCoeffDataDict"] = [
                new_holVarSpace + new_antiholVarSpace,
                otherDict,
            ]
        else:
            populate["compCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["realCoeffDataDict"] = [
                new_realVarSpace + new_imVarSpace,
                otherDict,
            ]

    return populate,new_realVarSpace,new_holVarSpace,new_antiholVarSpace,new_imVarSpace

def _shape_basis(basis,shape):
    if shape == 'symmetric':
        old_basis = dict(basis)
        new_basis = dict()
        for index, value in old_basis.items():
            new_index = tuple(sorted(index))
            if new_index in new_basis:
                new_basis[new_index] += value
            else:
                new_basis[new_index] = value
        return list(new_basis.items())
    if shape == 'skew':
        old_basis = dict(basis)
        new_basis = dict()
        for index, value in old_basis.items():
            permS, new_index = permSign(index,returnSorted=True)
            new_index = tuple(new_index)
            if new_index in new_basis:
                new_basis[new_index] += permS*value
            else:
                new_basis[new_index] = permS*value
        return list(new_basis.items())
    return basis


