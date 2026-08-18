from itertools import count

_card_uid = count()


class _vs_card:
    """
    Calling card for dgcv vector spaces (primarily for tensorProduct keys).

    Attributes
    ----------
    space : object or None
        The vector space this card stands for. ``None`` marks a dead card.
    root : _vs_card
        Card of the ambient space, or ``self`` for a root space.
    uid : int
        Monotonic identifier, for ordering and display.
    """

    __slots__ = ("space", "root", "uid")

    def __init__(self, space, root=None):
        self.space = space
        self.root = self if root is None else root
        self.uid = next(_card_uid)

    def __repr__(self):
        label = getattr(self.space, "label", None)
        tag = "dead" if self.space is None else (label or type(self.space).__name__)
        return f"<vs_card {self.uid}: {tag}>"


def card_root(card):
    """
    Resolve a key's card slot to its root card.
    """
    return card.root if type(card) is _vs_card else card
