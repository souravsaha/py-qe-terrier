import numpy as np
from iqg_learn import IdealQueryGeneration


class IdealQueryGenerationDFO(IdealQueryGeneration):
    def train_model(
        self,
        X,
        y,
        qid,
        term_list,
        tweak_magnitudes = [4.0, 2.0, 1.0, 0.5, 0.25],
        alpha = 2.0,
        beta = 64.0,
        gamma = 64.0,
        num_terms = 200
    ):

        oracle_rocchio = beta * np.mean(X[y == 1], axis=0) - gamma * np.mean(
            X[y == 0], axis=0
        )
        oracle_rocchio_qvec = IdealQueryGeneration.array_to_qvec(
            oracle_rocchio, term_list
        )

        oracle_rocchio_qvec.remove_non_positive_weights()
        oracle_rocchio_qvec.sort_by_weight()
        oracle_rocchio_qvec.trim(num_terms)

        current_ap = self.computeAP(qid, oracle_rocchio_qvec)

        for tm in tweak_magnitudes:
            for term, weight in oracle_rocchio_qvec.items():
                if current_ap >= 0.9:
                    break
                oracle_rocchio_qvec[term] = weight * (1 + tm)
                new_ap = self.computeAP(qid, oracle_rocchio_qvec)
                if new_ap > current_ap:
                    print(
                        f"{qid}\t{tm}\t{term}\t{current_ap}\t{new_ap} -- Accept tweak"
                    )
                    current_ap = new_ap
                else:
                    oracle_rocchio_qvec[term] = weight
                    print(
                        f"{qid}\t{tm}\t{term}\t{current_ap}\t{new_ap} -- Revert tweak"
                    )

        final_term_list = []
        final_arr = []
        for t, w in oracle_rocchio_qvec.items():
            final_term_list.append(t)
            final_arr.append(w)
        final_arr = np.array(final_arr)

        class TorchLinearRegWrapper:
            def __init__(self, coef_, term_list):
                self.coef_ = np.array([coef_])
                self.term_list = term_list

        return TorchLinearRegWrapper(final_arr, final_term_list)
