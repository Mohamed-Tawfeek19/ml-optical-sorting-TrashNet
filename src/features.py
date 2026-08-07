"""
Canonical feature extraction pipeline for the optical sorting project.

This module is the single source of truth for the 122-dimensional HSV+LBP
feature vector used by every classifier in this project:

    96 dims  3 x 32-bin normalised HSV colour histograms (H, S, V)
    26 dims  uniform LBP texture histogram (radius=3, P=24)
    ---
   122 dims

Every notebook that extracts features imports from here, so the training
features (notebook 02) and the evaluation features (notebooks 06, 07, 12, 13
and the demo) are guaranteed to come from identical code.

Implementation notes
--------------------
`resize` uses Pillow's default BICUBIC filter and the LBP greyscale is derived
via an HSV -> BGR -> GREY conversion. Both are deliberate: they define the
feature space the saved models were fitted on. Substituting LANCZOS resizing or
a direct RGB -> GREY conversion shifts every feature slightly and moves test
accuracy by several tenths of a point, so they are kept exactly as trained.
"""

import os

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

# scikit-learn orders classes alphabetically internally. Anything that labels a
# confusion matrix, a classification report or a per-class table must use this
# order, never the CLASSES order above.
CLASS_ORDER = sorted(CLASSES)

IMAGE_SIZE = (224, 224)
HISTOGRAM_BINS = 32
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_BINS = LBP_N_POINTS + 2
N_FEATURES = 3 * HISTOGRAM_BINS + LBP_BINS

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def preprocess_pil(image):
    """Resize a PIL RGB image to IMAGE_SIZE and convert it to HSV."""
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image)

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    return cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)


def preprocess_image(image_path):
    """Load an image from disk, resize it, and convert it to HSV."""
    return preprocess_pil(Image.open(image_path).convert('RGB'))


def extract_colour_histogram(img_hsv):
    """Normalised 32-bin histogram per HSV channel, concatenated to 96 dims."""
    features = []

    for channel in range(3):
        histogram = cv2.calcHist(
            [img_hsv],
            [channel],
            None,
            [HISTOGRAM_BINS],
            [0, 256]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()
        features.extend(histogram)

    return np.array(features)


def extract_lbp_features(img_hsv):
    """Uniform LBP texture descriptor as a 26-bin normalised density histogram."""
    img_gray = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(img_gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')

    histogram, _ = np.histogram(
        lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True
    )

    return histogram


def features_from_hsv(img_hsv):
    """Combine colour and texture descriptors into the 122-dim feature vector."""
    return np.concatenate([
        extract_colour_histogram(img_hsv),
        extract_lbp_features(img_hsv),
    ])


def extract_features(image_path):
    """Full 122-dim feature vector for an image on disk."""
    return features_from_hsv(preprocess_image(image_path))


def extract_features_from_image(image, perturb_fn=None):
    """
    Full 122-dim feature vector for an in-memory PIL image.

    `perturb_fn` is applied to the image at its original resolution, before
    resizing, so that perturbations model sensor-side degradation rather than
    an artefact introduced after downsampling.
    """
    if perturb_fn is not None:
        image = perturb_fn(image)

    return features_from_hsv(preprocess_pil(image))


def list_class_images(dataset_path, class_name):
    """Sorted list of image paths for one class. Sorted for cross-platform determinism."""
    class_folder = os.path.join(dataset_path, class_name)
    filenames = sorted(
        f for f in os.listdir(class_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    return [os.path.join(class_folder, f) for f in filenames]


def load_dataset(dataset_path, perturb_fn=None, classes=None):
    """
    Extract features for every image in the dataset, optionally under a perturbation.

    Returns (features, labels) as arrays of shape (n_images, 122) and (n_images,).
    """
    classes = CLASSES if classes is None else classes

    features_list = []
    labels_list = []

    for class_name in classes:
        for image_path in list_class_images(dataset_path, class_name):
            image = Image.open(image_path).convert('RGB')
            features_list.append(extract_features_from_image(image, perturb_fn))
            labels_list.append(class_name)

    return np.array(features_list), np.array(labels_list)
