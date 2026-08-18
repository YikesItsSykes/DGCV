from collections import Counter

from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..dgcv_core.spaces.spaces import card_root
from .shapes import _shape_filter


def _card_root_map(cards):
    return {card_root(card): card for card in cards}


def _promotion_targets(cards, _amb_prom):
    groups = {}
    for card in cards:
        root = card_root(card)
        group = groups.get(root)
        if group is None:
            groups[root] = [card]
        else:
            group.append(card)
    targets = set()
    for root, group in groups.items():
        if _amb_prom or len(group) > 1:
            for card in group:
                if card is not root and card.space is not None:
                    targets.add(card)
    return targets


def _resolve_binary_promotions(map_a, map_b):
    targets_a = set()
    targets_b = set()
    for root, card_a in map_a.items():
        card_b = map_b.get(root)
        if card_b is None or card_a == card_b:
            continue
        if card_a is not root and card_a.space is not None:
            targets_a.add(card_a)
        if card_b is not root and card_b.space is not None:
            targets_b.add(card_b)
    return targets_a, targets_b


def _promote_keys(coeff_dict, targets):
    out = Counter()
    for key, value in coeff_dict.items():
        positions = [c for c, factor in enumerate(key) if factor[2] in targets]
        if not positions:
            out[key] += value
            continue
        terms = [(key, value)]
        for c in positions:
            new_terms = []
            for k, v in terms:
                idx, valence, card = k[c]
                root = card.root
                head, tail = k[:c], k[c + 1 :]
                for pos, co in enumerate(card.space.basis[idx].ambient_rep.coeffs):
                    if _scalar_is_zero(co):
                        continue
                    new_terms.append((head + ((pos, valence, root),) + tail, co * v))
            terms = new_terms
        for k, v in terms:
            out[k] += v
    return out


def _merge_normalized(lhs, rhs, sign, shape, accumulate):
    processed = dict(lhs.coeff_dict)
    touched = []
    if sign > 0:
        for key, val in rhs.coeff_dict.items():
            if key in processed:
                processed[key] = processed[key] + val
                touched.append(key)
            else:
                processed[key] = val
    else:
        for key, val in rhs.coeff_dict.items():
            if key in processed:
                processed[key] = processed[key] - val
                touched.append(key)
            else:
                processed[key] = -val
    dropped = False
    for key in touched:
        if _scalar_is_zero(processed[key]):
            del processed[key]
            dropped = True
    scalar = processed.get(tuple(), None)
    if scalar is not None and _scalar_is_zero(scalar):
        del processed[tuple()]
        dropped = True
    seen = set()
    cards = []
    if dropped:
        for key in processed:
            for factor in key:
                card = factor[2]
                if card not in seen:
                    seen.add(card)
                    cards.append(card)
    else:
        for card in lhs.vs_id + rhs.vs_id:
            if card not in seen:
                seen.add(card)
                cards.append(card)
    targets = _promotion_targets(cards, False) if cards else set()
    if targets:
        processed = dict(_promote_keys(processed, targets))
        vs_id = []
        promoted = set()
        for card in cards:
            card = card_root(card) if card in targets else card
            if card not in promoted:
                promoted.add(card)
                vs_id.append(card)
        vs_id = tuple(vs_id)
        if shape == "skew" or shape == "symmetric" or callable(shape):
            processed = _shape_filter(processed, shape=shape, accumulate=accumulate)
        dropped = True
    else:
        vs_id = tuple(cards)
    if not processed:
        return {tuple(): 0}, 0, -1, tuple()
    if dropped:
        max_degree = 0
        min_degree = -1
        for key in processed:
            deg = len(key)
            if deg > max_degree:
                max_degree = deg
            if min_degree < 0 or deg < min_degree:
                min_degree = deg
    else:
        max_degree = max(lhs.max_degree, rhs.max_degree)
        min_degree = min(lhs.min_degree, rhs.min_degree)
    return processed, max_degree, min_degree, vs_id
