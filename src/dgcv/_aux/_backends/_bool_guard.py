import collections
import contextlib
import operator
import traceback

_ALLOWLIST = frozenset()


class bool_guard_report:
    def __init__(self, hits, wrong, samples, allowlist):
        self.hits = hits
        self.wrong = wrong
        self.samples = samples
        self.allowlist = frozenset(allowlist)

    @property
    def total(self):
        return sum(self.hits.values())

    @property
    def offending_sites(self):
        return sorted(s for s in self.hits if s not in self.allowlist)

    @property
    def wrong_sites(self):
        return sorted(self.wrong)

    def format(self):
        lines = []
        for site in sorted(self.hits, key=lambda s: -self.hits[s]):
            flag = ""
            if site in self.wrong:
                flag = "  <== WRONG ANSWERS (%d)" % self.wrong[site]
            elif site in self.allowlist:
                flag = "  (allowlisted)"
            lines.append("  %6d  %s%s" % (self.hits[site], site, flag))
            if site in self.samples:
                lines.append("          e.g. %s" % self.samples[site])
        return "\n".join(lines)


@contextlib.contextmanager
def detect_unstable_bools(allowlist=_ALLOWLIST):
    hits = collections.Counter()
    wrong = collections.Counter()
    samples = {}

    try:
        import sage.symbolic.relation as _rel
    except Exception:
        yield bool_guard_report(hits, wrong, samples, allowlist)
        return

    from ._symbolic_router import _scalar_is_zero

    original = _rel.test_relation_maxima

    def hooked(relation, *args, **kwargs):
        frames = [
            f
            for f in traceback.extract_stack()
            if "/dgcv/" in f.filename and not f.filename.endswith("_bool_guard.py")
        ]
        if frames:
            site = "%s:%d(%s)" % (
                frames[-1].filename.split("/dgcv/")[-1],
                frames[-1].lineno,
                frames[-1].name,
            )
        else:
            site = "(non-dgcv)"

        result = original(relation, *args, **kwargs)
        hits[site] += 1

        try:
            router_zero = _scalar_is_zero(relation.lhs() - relation.rhs())
            op = relation.operator()
            raw = bool(result)
            if (op is operator.ne and raw is False and not router_zero) or (
                op is operator.eq and raw is False and router_zero
            ):
                wrong[site] += 1
                samples.setdefault(site, str(relation)[:110])
        except Exception:
            pass

        return result

    _rel.test_relation_maxima = hooked
    try:
        yield bool_guard_report(hits, wrong, samples, allowlist)
    finally:
        _rel.test_relation_maxima = original
