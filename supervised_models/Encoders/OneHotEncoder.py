import numpy as np


class OneHotEncoder(object):

    def encode_data(self, data: np.ndarray) -> np.ndarray:
        """
        Generates a one-hot encoding representation for an entire set of observations.

        Args:
            data: Array of arrays of identifiers

        Returns:
            np.ndarray: Numpy ndarray of one-hot-encoded representations.
        """
        all_options: set = set()
        for entry in data:
            all_options.update(entry)

        all_encodings: list = []
        for entry in data:
            one_hot = np.zeros(len(all_options))
            one_hot[[i for i, option in enumerate(all_options) if option in entry]] = 1
            all_encodings.append(one_hot)

        return np.asarray(all_encodings)





