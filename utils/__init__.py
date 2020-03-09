import numpy as np


def select_prototypes(X, y, n_prototypes):
        '''
            Selects prototypes from each class according to n_prototypes.
            If n_prototypes is None, selects 10% of each class as 
            prototypes.
        '''

        labels = np.unique(y)
        selected_X = []
        selected_y = []
        for label in labels:
            idx = np.where(y == label)[0]
            if not n_prototypes:
                n_prots = int(0.1 * len(idx))
            else:
                n_prots = n_prototypes[label]
                if type(n_prots) is float:
                    n_prots = int(n_prots * len(idx))
            seleted_idx = np.random.choice(idx, n_prots, replace=False)
            selected_X.append(X[seleted_idx])
            selected_y.append(y[seleted_idx])


        return (
            np.vstack(selected_X),
            np.hstack(selected_y)
        )