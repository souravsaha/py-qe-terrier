from collections import OrderedDict, defaultdict
from typing import Union
import csv
import os


class QueryVector:
    def __init__(self, vector: Union[dict[str, float], None] = None) -> None:
        if vector is None:
            vector = defaultdict(float)
        self.vector = vector

    def __setitem__(self, term: str, weight: float) -> None:
        self.vector[term] = weight

    def __getitem__(self, term: str) -> float:
        return self.vector[term]

    def __contains__(self, term) -> bool:
        return term in self.vector

    def __len__(self) -> int:
        return len(self.vector)

    def __iter__(self):
        return self.vector.__iter__()

    def items(self):
        return self.vector.items()

    def values(self):
        return self.vector.values()

    def keys(self):
        return self.vector.keys()

    def sort_by_weight(self, reverse=True) -> None:
        self.vector = OrderedDict(
            sorted(self.vector.items(), key=lambda x: x[1], reverse=reverse)
        )

    def trim(self, num_keep_terms: int) -> None:
        self.vector = OrderedDict(list(self.vector.items())[:num_keep_terms])

    def remove_non_positive_weights(self) -> None:
        new_vector = OrderedDict()
        for term, weight in self.vector.items():
            if weight <= 0:
                continue
            new_vector[term] = weight
        self.vector = new_vector

    def remove_zero_weights(self) -> None:
        new_vector = OrderedDict()
        for term, weight in self.vector.items():
            if weight == 0:
                continue
            new_vector[term] = weight
        self.vector = new_vector

    def remove_lt_threshold(self, threshold):
        new_vector = OrderedDict()
        for term, weight in self.vector.items():
            if weight < threshold:
                continue
            new_vector[term] = weight
        self.vector = new_vector

    def store_txt(self, qid, store_path: str, append=True) -> None:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.sort_by_weight()  # always sort according to weight before storing
        with open(store_path, "a" if append else "w") as store_file:
            writer = csv.writer(store_file, delimiter="\t")
            for term, weight in self.vector.items():
                writer.writerow([qid, term, weight])
