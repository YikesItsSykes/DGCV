class _tp_weights:
    def _compute_weights(
        self,
        test_weights=None,
        bound_number_of_weights=None,
        use_aggresive_weight_search=False,
    ):
        """Lazily compute weights for all terms in coeff_dict for each weight vector of the parent vector space or in a given list `test_weights`."""
        if test_weights is None:
            default_test = True
            test_weights = {}
            mw = None
            changed = None
            for card in self.vs_id:
                gl = card.space.grading
                if mw is None:
                    if bound_number_of_weights is None:
                        mw = len(gl)
                    else:
                        mw = min(bound_number_of_weights, len(gl))
                    changed = False
                elif mw > len(gl):
                    mw = len(gl)
                    if changed is False:
                        changed = True
                test_weights[card] = gl
            if changed is True:
                for card in self.vs_id:
                    test_weights[card] = test_weights[card][:mw]
        else:
            default_test = False
            if isinstance(test_weights, (list, tuple)):
                mw = len(test_weights)
                test_weights = {
                    card: test_weights for card in self.vs_id
                }  # supporting old format
            else:
                mw = len(next(iter(test_weights.values())))
        if default_test is False or self._weights is None:
            weight_dict = {}
            for (
                key,
                _,
            ) in (
                self.coeff_dict.items()
            ):  # algo requires valence 1 for vec and 0 for covec
                weight_list = []
                for w_idx in range(mw):
                    weight = 0
                    for index, valence, card in key:
                        twL1 = test_weights[card]
                        twL2 = twL1[w_idx]
                        twL3 = twL2[index]
                        weight += twL3 * (valence * 2 - 1)
                    weight_list.append(weight)
                weight_dict[key] = tuple(weight_list)
            if default_test is True:
                self._weights = weight_dict
            else:
                return weight_dict

        return self._weights

    def check_element_weight(
        self,
        test_weights=None,
        _return_mixed_weight_list=False,
        bound_number_of_weights=None,
    ):
        return self.compute_weight(
            test_weights=test_weights,
            _return_mixed_weight_list=_return_mixed_weight_list,
            bound_number_of_weights=bound_number_of_weights,
        )

    def compute_weight(
        self,
        test_weights=None,
        _return_mixed_weight_list=False,
        bound_number_of_weights=None,
    ):  ###!!! review
        weights = list(
            set(
                (
                    self._compute_weights(
                        test_weights=test_weights,
                        bound_number_of_weights=bound_number_of_weights,
                    )
                ).values()
            )
        )
        if _return_mixed_weight_list is True:
            return weights
        if len(weights) == 1:
            return weights[0]
        else:
            return "NoW"

    def get_weighted_components(self, weight_list, test_weights=None):
        """
        Return a new tensorProduct with components matching the given weight_list.

        Parameters:
        - weight_list: A list or tuple of weights to match against.

        Returns:
        - A new tensorProduct with a filtered coeff_dict.
        """
        wd = self._compute_weights(test_weights=test_weights)

        filtered_coeff_dict = {
            key: value
            for key, value in self.coeff_dict.items()
            if wd[key] == tuple(weight_list)
        }

        return tensorProduct(filtered_coeff_dict)
