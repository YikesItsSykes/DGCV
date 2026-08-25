from __future__ import annotations

import numbers
import random

from ..._aux._utilities._config import (
    dgcv_warning,
)
from ...core.arrays import freeze_matrix, matrix_dgcv
from .algebra_classifications import (
    RealFormReport,
    real_form_by_label,
    real_form_candidates,
)
from .util import (
    compute_root_length_profile,
    decompose_semisimple_algebra,
    fast_rank,
    killingForm,
)


def approximate_rank(
    target_alg,
    check_semisimple=False,
    assume_semisimple=False,
    _use_cache=False,
    surface_singularities=False,
    simplify_singularities=None,
):
    if target_alg.dimension == 0:
        target_alg._rank_approximation = 0
        if surface_singularities:
            return 0, []
        return 0
    if check_semisimple is True:
        ssc = target_alg.is_semisimple()
        if ssc is True:
            assume_semisimple = True
        elif assume_semisimple is True:
            print(
                "approximate_rank received parameters `check_semisimple=True` and `assume_semisimple=True`, but the semisimple check returned false. The algorithm is proceeding with the `assume_semisimple` logic applied, but this is likely not wanted, and should be prevented by setting those parameters differently. Note, just setting `check_semisimple=True` is enough to use optimized algorithms in the event that the semisimple check returns true, whereas `assume_semisimple` should only be used in applications where forgoing the semisimple check entirely is wanted."
            )
    if _use_cache and target_alg._rank_approximation is not None:
        if surface_singularities:
            return target_alg._rank_approximation, []
        return target_alg._rank_approximation
    power_bound = (
        1
        if (assume_semisimple or target_alg._is_semisimple_cache is True)
        else target_alg.dimension
    )
    get_slice = target_alg._structure_data_slice
    elem = matrix_dgcv(
        get_slice(0), shape=target_alg.structureData.shape
    )  # test element
    bound = max(100, 10 * target_alg.dimension)
    for idx in range(1, target_alg.dimension):
        elem2 = get_slice(idx)
        elem += random.randint(1, bound) * matrix_dgcv(
            elem2, shape=target_alg.structureData.shape
        )
    divisors = []
    test_mat, test_rank = 1, target_alg.dimension
    for _ in range(power_bound):
        test_mat = test_mat * elem
        rank_result = fast_rank(
            test_mat,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
        )
        if surface_singularities:
            rank, new_divisors = rank_result
            divisors += new_divisors
        else:
            rank = rank_result
        if test_rank == rank:
            break
        test_rank = rank
    rank = target_alg.dimension - rank
    if not isinstance(rank, numbers.Integral):
        dgcv_warning(
            "`approximate_rank` failed"
            f"({rank}); the cached rank approximation was left unchanged."
        )
    elif (
        not isinstance(target_alg._rank_approximation, numbers.Integral)
        or target_alg._rank_approximation > rank
    ):
        target_alg._rank_approximation = rank
    if surface_singularities:
        return target_alg._rank_approximation, divisors
    return target_alg._rank_approximation


def _require_real_semisimple(
    target_alg,
    method_name,
    assume_semisimple=False,
    assume_simple=False,
    assume_real=False,
    assume_Lie_algebra=False,
):
    if assume_Lie_algebra is False:
        target_alg._require_lie_algebra(method_name)
    if not (assume_real or target_alg.base_field == "real"):
        raise ValueError(
            f"{method_name} reads the algebra as a real Lie algebra, but its "
            "`base_field` is 'complex'. Set `base_field='real'` on the "
            "algebra, or pass `assume_real=True`. Real structure constants "
            "alone do not settle this, since a complex Lie algebra can be "
            "presented in a basis over which they are real."
        ) from None
    if not (assume_simple or assume_semisimple or target_alg.is_semisimple()):
        raise ValueError(
            f"{method_name} can only be applied to semisimple Lie algebras."
        ) from None


def _frozen_killing_form(target_alg, assume_Lie_algebra=False):
    if target_alg._killing_form is None:
        target_alg._killing_form = freeze_matrix(
            killingForm(target_alg, assume_Lie_algebra=assume_Lie_algebra)
        )
    else:
        target_alg._killing_form = freeze_matrix(target_alg._killing_form)
    return target_alg._killing_form


def killing_inertia(
    target_alg,
    signature_only: bool = False,
    assume_semisimple=False,
    assume_simple=False,
    assume_real=False,
    assume_Lie_algebra=False,
):
    """
    Sylvester inertia of the Killing form.

    Parameters
    ----------
    signature_only : bool, default False
        Return the signature `p - n` in place of the full inertia.
    assume_semisimple : bool, default False
        Skip the semisimplicity check.
    assume_simple : bool, default False
        Skip the semisimplicity check. Does not imply `assume_real`.
    assume_real : bool, default False
        Skip the check that the structure constants are real.
    assume_Lie_algebra : bool, default False
        Skip the Lie algebra check.

    Returns
    -------
    tuple of int or int
        `(p, n, z)`, the counts of positive, negative, and zero
        eigenvalues, or `p - n` when `signature_only` is set.

    Raises
    ------
    IndeterminateSignError
        If a pivot sign cannot be certified, as happens when the
        structure constants carry parameters.
    """
    target_alg._require_real_semisimple(
        "killing_inertia",
        assume_semisimple=assume_semisimple,
        assume_simple=assume_simple,
        assume_real=assume_real,
        assume_Lie_algebra=assume_Lie_algebra,
    )
    kf = target_alg._frozen_killing_form(assume_Lie_algebra=assume_Lie_algebra)
    if signature_only is True:
        return kf.symmetric_signature()
    return kf.symmetric_inertia()


def is_compact_real_form(
    target_alg,
    assume_semisimple=False,
    assume_simple=False,
    assume_real=False,
    assume_Lie_algebra=False,
):
    """
    Tests whether the algebra is a compact real form.

    Parameters
    ----------
    assume_semisimple : bool, default False
        Skip the semisimplicity check.
    assume_simple : bool, default False
        Skip the semisimplicity check. Does not imply `assume_real`.
    assume_real : bool, default False
        Skip the check that the structure constants are real.
    assume_Lie_algebra : bool, default False
        Skip the Lie algebra check.

    Returns
    -------
    bool

    Notes
    -----
    Equivalent to negative definiteness of the Killing form, so the
    verdict is meaningful for compact semisimple algebras and not only
    simple ones.
    """
    target_alg._require_real_semisimple(
        "is_compact_real_form",
        assume_semisimple=assume_semisimple,
        assume_simple=assume_simple,
        assume_real=assume_real,
        assume_Lie_algebra=assume_Lie_algebra,
    )
    kf = target_alg._frozen_killing_form(assume_Lie_algebra=assume_Lie_algebra)
    return kf.is_negative_definite_symmetric()


def is_split_real_form(
    target_alg,
    assume_semisimple=False,
    assume_simple=False,
    assume_real=False,
    assume_Lie_algebra=False,
):
    """
    Tests whether the algebra is a split real form.

    Parameters
    ----------
    assume_semisimple : bool, default False
        Skip the semisimplicity check.
    assume_simple : bool, default False
        Skip the semisimplicity check. Does not imply `assume_real`.
    assume_real : bool, default False
        Skip the check that the structure constants are real.
    assume_Lie_algebra : bool, default False
        Skip the Lie algebra check.

    Returns
    -------
    bool

    Notes
    -----
    Compares the Killing signature against `approximate_rank`, which
    bounds the rank from above, so a False verdict can be an artifact of
    an overestimated rank.
    """
    target_alg._require_real_semisimple(
        "is_split_real_form",
        assume_semisimple=assume_semisimple,
        assume_simple=assume_simple,
        assume_real=assume_real,
        assume_Lie_algebra=assume_Lie_algebra,
    )
    kf = target_alg._frozen_killing_form(assume_Lie_algebra=assume_Lie_algebra)
    rank = target_alg.approximate_rank(_use_cache=True, assume_semisimple=True)
    return kf.symmetric_signature() == rank


def _certify_centroid_type(target_alg):
    if target_alg._centroid_type is not None:
        return target_alg._centroid_type
    try:
        _, types = decompose_semisimple_algebra(
            target_alg, assume_semisimple=True, return_centroid_types=True
        )
    except Exception:
        return None
    if len(types) == 1 and types[0] is not None:
        target_alg._centroid_type = types[0]
    return target_alg._centroid_type


def root_length_profile(
    target_alg,
    assume_simple=False,
    assume_Lie_algebra=False,
    attempts=4,
    _use_cache=True,
):
    """
    Killing-normalized squared root lengths of the complexification.

    Parameters
    ----------
    assume_simple : bool, default False
        Skip the simplicity check.
    assume_Lie_algebra : bool, default False
        Skip the Lie algebra check.
    attempts : int, default 4
        Number of random elements tried before giving up.
    _use_cache : bool, default True

    Returns
    -------
    RealFormReport or None
        `None` when no attempt produced a Cartan subalgebra with a
        rational length spectrum, e.g., when parameters are present.
    """
    if _use_cache and target_alg._root_length_profile_cache is not None:
        return target_alg._root_length_profile_cache
    if assume_Lie_algebra is False:
        target_alg._require_lie_algebra("root_length_profile")
    if assume_simple is False and not target_alg.is_simple():
        raise ValueError(
            "root_length_profile can only be applied to simple Lie algebras."
        ) from None
    profile = compute_root_length_profile(target_alg, attempts=attempts)
    if profile is not None:
        target_alg._root_length_profile_cache = profile
        target_alg._rank_approximation = profile.rank
    return profile


def identify_complex_type(
    target_alg,
    assume_simple=False,
    assume_Lie_algebra=False,
    attempts=4,
    _use_cache=True,
):
    """
    Dynkin type of the complexification.

    Parameters
    ----------
    assume_simple : bool, default False
        Skip the simplicity check.
    assume_Lie_algebra : bool, default False
        Skip the Lie algebra check.
    attempts : int, default 4
        Number of random elements tried before giving up.
    _use_cache : bool, default True

    Returns
    -------
    str or None
        A tag such as `'B3'`, or `'A1+A1'` for a realification. `None`
        when the measurement failed or matched no type.
    """
    profile = target_alg.root_length_profile(
        assume_simple=assume_simple,
        assume_Lie_algebra=assume_Lie_algebra,
        attempts=attempts,
        _use_cache=_use_cache,
    )
    return None if profile is None else profile.complex_type


def _narrow_by_complex_type(target_alg, report):
    if not report.candidates:
        return report
    try:
        profile = target_alg.root_length_profile(
            assume_simple=True, assume_Lie_algebra=True
        )
    except Exception:
        return report
    if profile is None or profile.complex_type is None:
        return report
    candidates = report.candidates
    if profile.rank != report.rank:
        candidates, _ = real_form_candidates(
            report.dimension,
            profile.rank,
            report.signature,
            report.absolutely_simple,
        )
    kept = tuple(
        record for record in candidates if record.complex_type == profile.complex_type
    )
    if not kept:
        return report
    return RealFormReport(
        candidates=kept,
        certain=len(kept) == 1,
        dimension=report.dimension,
        rank=profile.rank,
        signature=report.signature,
        absolutely_simple=report.absolutely_simple,
        rank_is_probabilistic=False,
        source=report.source,
    )


def _finalize_real_form(target_alg, report, tier2):
    if tier2 is True and report.certain is False and report.source == "computed":
        report = target_alg._narrow_by_complex_type(report)
    target_alg._real_form_cache = report
    return report


def identify_real_form(
    target_alg,
    assume_simple=False,
    assume_real=False,
    assume_Lie_algebra=False,
    certify_centroid=None,
    tier2=False,
    _use_cache=True,
):
    """
    Identifies which real form of a simple Lie algebra this is.

    Parameters
    ----------
    assume_simple : bool, default False
        Skip the semisimplicity and simplicity checks. Does not imply
        `assume_real`.
    assume_real : bool, default False
        Skip the check that the structure constants are real.
    assume_Lie_algebra : bool, default False
        Skip the Lie algebra check.
    certify_centroid : bool, optional
        Whether to compute the centroid verdict separating an absolutely
        simple algebra from a realification. The default computes it only
        when the Killing signature vanishes, which is the only case where
        it changes the candidate set.
    tier2 : bool, default False
        Whether to resolve a candidate set by measuring the root lengths
        of the complexification. Opt-in, since it costs a Cartan
        subalgebra and a Casimir operator.
    _use_cache : bool, default True

    Returns
    -------
    RealFormReport
        Candidate real forms together with the invariants they were drawn
        from. More than one candidate means the invariants collide and
        Tier 2 identification is needed to separate them.

    Notes
    -----
    A name, or sequence of names, seeded in
    `_educed_properties['real_form']` short-circuits the computation.
    The rank is supplied by `approximate_rank` and is probabilistic
    unless Tier 2 ran and certified it.
    """
    if _use_cache and target_alg._real_form_cache is not None:
        return target_alg._finalize_real_form(target_alg._real_form_cache, tier2)

    if target_alg.dimension == 0:
        return target_alg._finalize_real_form(
            RealFormReport(
                candidates=(),
                certain=False,
                dimension=0,
                rank=0,
                signature=0,
                absolutely_simple=None,
                rank_is_probabilistic=False,
                source="computed",
            ),
            tier2,
        )

    asserted = target_alg._educed_properties.get("real_form", None)
    if isinstance(asserted, str):
        asserted = (asserted,)
    if asserted:
        records = []
        unresolved = []
        for label in asserted:
            record = real_form_by_label(label, target_alg.dimension)
            if record is None:
                unresolved.append(label)
            else:
                records.append(record)
        if unresolved:
            dgcv_warning(
                "`identify_real_form` did not recognize the asserted real form "
                f"label(s) {unresolved}, so they were skipped."
            )
        if records:
            ranks = {record.rank for record in records}
            signatures = {record.signature for record in records}
            realifications = {record.is_realification for record in records}
            return target_alg._finalize_real_form(
                RealFormReport(
                    candidates=tuple(records),
                    certain=len(records) == 1,
                    dimension=target_alg.dimension,
                    rank=ranks.pop() if len(ranks) == 1 else None,
                    signature=signatures.pop() if len(signatures) == 1 else None,
                    absolutely_simple=not realifications.pop()
                    if len(realifications) == 1
                    else None,
                    rank_is_probabilistic=False,
                    source="asserted",
                ),
                tier2,
            )

    target_alg._require_real_semisimple(
        "identify_real_form",
        assume_simple=assume_simple,
        assume_real=assume_real,
        assume_Lie_algebra=assume_Lie_algebra,
    )
    if not (assume_simple or target_alg.is_simple()):
        raise ValueError(
            "identify_real_form can only be applied to simple Lie algebras."
        ) from None

    signature = target_alg.killing_inertia(
        signature_only=True,
        assume_simple=True,
        assume_real=True,
        assume_Lie_algebra=True,
    )
    rank = target_alg.approximate_rank(_use_cache=True, assume_semisimple=True)
    if certify_centroid is True or (certify_centroid is None and signature == 0):
        target_alg._certify_centroid_type()
    absolutely_simple = {"real": True, "complex": False}.get(
        target_alg._centroid_type, None
    )
    candidates, certain = real_form_candidates(
        target_alg.dimension, rank, signature, absolutely_simple
    )
    return target_alg._finalize_real_form(
        RealFormReport(
            candidates=candidates,
            certain=certain,
            dimension=target_alg.dimension,
            rank=rank,
            signature=signature,
            absolutely_simple=absolutely_simple,
            rank_is_probabilistic=True,
            source="computed",
        ),
        tier2,
    )
