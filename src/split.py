"""
Canonical stratified 70/15/15 split for the optical sorting project.

`make_split` is the function notebook 15 defines inline, which is itself the
split notebooks 04 and 05 perform in two `train_test_split` calls. It is lifted
here verbatim so that notebooks added after 15 can partition against exactly the
same definition instead of copying it. Notebook 15 is left as it is.

Why an index variant
--------------------
Every classifier before notebook 16 consumes the 122-dim feature matrix, so
splitting the matrix directly is enough. A CNN consumes raw images, and its
partition has to be the *same images* as the feature matrix partition or none of
the comparisons hold.

`split_indices` solves that by splitting positional indices instead of data.
`sklearn.model_selection.train_test_split` derives its permutation from
`random_state`, the sample count and the `stratify` labels only, never from the
contents of what it is splitting, so splitting `arange(n)` under the same seed
and the same label array reproduces the partition of any array indexed by it.
Notebook 16 asserts this rather than assuming it: it indexes the committed
feature matrix by the returned test indices, scores the committed Random Forest
pickle on the result, and checks the confusion matrix against the committed
`results/rf_confusion_matrix.npy`.

The image path list must be built in the same order the feature matrix was, that
is `src.features.CLASSES` order with `list_class_images` supplying the
per-class filenames. Any other order silently pairs each index with the wrong
image.
"""

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.176   # 0.176 of the remaining 85% gives 15% of the whole


def make_split(features, labels, seed=RANDOM_STATE):
    """Stratified 70/15/15 split.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_indices(labels, seed=RANDOM_STATE):
    """The same split expressed as positional indices into `labels`.

    Returns (train_idx, val_idx, test_idx). Indexing any array that is aligned
    with `labels` by these gives exactly the partitions `make_split` returns for
    that array.
    """
    indices = np.arange(len(labels))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=labels[train_val_idx]
    )

    return train_idx, val_idx, test_idx
