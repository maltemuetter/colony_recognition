import cv2
from scipy import ndimage
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import binary_closing, disk, binary_dilation
from .objects import Obj
import warnings


class ColonyRecognition:
    def __init__(
        self,
        d_im: np.ndarray,
        max_px=5000,
        d=3,
        low_thres=40,
        dist_coef=0.6,
        px_min=15,
    ):
        self.image = d_im
        self.value_thres = self.value_thresholding(self.image, low_thres)
        self.closed_value_mask = self.fill_holes(self.value_thres, d)
        self.distance_thres_image, self.dist_img = self.distance_thresholding(
            self.closed_value_mask, dist_coef
        )
        self.labels = self.apply_watershed(self.dist_img, self.distance_thres_image)
        self.size_filtered_labels = self.size_filter_labels(self.labels, max_px=max_px)
        self.find_objects(px_min)
        self.corrected = False

    @staticmethod
    def value_thresholding(image, low_thresh, show=False):
        _, thresh = cv2.threshold(image, low_thresh, 255, cv2.THRESH_BINARY)
        if show:
            show_difference(image, thresh)
        return thresh

    @staticmethod
    def fill_holes(img, d, show=False):
        # Define structuring element for the closing operation
        selem = disk(d)
        # Perform morphological closing operation
        closed_thresh_mask = binary_closing(img, selem)
        if show:
            show_difference(img, closed_thresh_mask)
        return closed_thresh_mask

    @staticmethod
    def distance_thresholding(im_mask, dist_coef, show=False):
        # Get distance to next zero. Thin walls will be very close to zero
        D_closed = ndimage.distance_transform_edt(im_mask)
        # optimize minimum distance
        count, distance = np.histogram(D_closed)
        mean_dist = np.sum(count[2:] * distance[3:]) / np.sum(count[2:])
        d_thres = mean_dist * dist_coef
        # Filter out values very close to zero
        _, D_thresh = cv2.threshold(D_closed, d_thres, 255, cv2.THRESH_BINARY)
        # Restore colony edges
        selem = disk(d_thres)
        restored_edges = binary_dilation(D_thresh, selem)
        restored_image = (
            np.logical_or(restored_edges, D_thresh > 0).astype(np.uint8) * 255
        )
        if show:
            show_difference(im_mask, restored_image)
        return restored_image, ndimage.distance_transform_edt(restored_image)

    @staticmethod
    def apply_watershed(dist_img, dist_thres_image):
        # Compute the markers using connected components
        markers, _ = ndimage.label(dist_thres_image)
        # Apply the watershed algorithm
        labels = watershed(
            -dist_img, markers, mask=dist_thres_image, watershed_line=True
        )
        return labels

    def size_filter_labels(self, labels, max_px=5000):
        # Remove small objects
        unique, counts = np.unique(labels, return_counts=True)
        min_px = np.median(counts[1:]) / 4
        size_filtered_labels = np.zeros_like(labels)
        for value, count in zip(unique, counts):
            if (count >= min_px) & (count <= max_px):
                size_filtered_labels[labels == value] = value
        return size_filtered_labels

    def show(self, features, figsize=(16, 5), colonies=False):
        n_features = len(features) + colonies
        n_cols = n_features
        _, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        for i, feature in enumerate(features):
            try:
                feature_data = getattr(self, feature)
                if isinstance(feature_data, str):
                    warnings.warn(
                        f"Skipping feature '{feature}' because it's not an image."
                    )
                    continue
                if len(feature_data.shape) == 2:
                    axes[i].imshow(feature_data, cmap="gray")
                else:
                    axes[i].imshow(feature_data)
                axes[i].set_title(feature)
                axes[i].axis("off")
            except AttributeError:
                warnings.warn(f"Skipping unknown feature '{feature}'.")
                continue
        if colonies:
            self.show_colonies(ax=axes[-1])
        plt.show()

    def find_objects(self, px_min):
        self.object_nums = []
        self.objects = []
        self.colonies = []
        for o in np.unique(self.labels)[1:]:
            obj = Obj(self.labels, o, px_min)
            if obj.use:
                self.objects.append(obj)
                self.object_nums.append(o)
                self.colonies += obj.colonies

    def plot_labels(self, labels):
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        if n_labels > 1:
            colormap = plt.cm.get_cmap("nipy_spectral", n_labels)
            colors = [colormap(i) for i in range(n_labels)]
        else:
            colors = ["black"]

        label_colored_image = np.zeros((*labels.shape, 3), dtype=float)
        for label, color in zip(unique_labels, colors):
            if label == 0:
                continue
            label_colored_image[labels == label] = color[:3]

        plt.figure(figsize=(10, 6))
        plt.imshow(label_colored_image)
        plt.axis("off")
        plt.title("Colored Labels")
        plt.show()

    @staticmethod
    def show_difference(original, modified):
        diff_img = np.zeros((*original.shape, 3), dtype=np.uint8)
        became_zero = (original > 0) & (modified == 0)
        diff_img[became_zero] = [255, 0, 0]  # Red
        not_zero_anymore = (original == 0) & (modified > 0)
        diff_img[not_zero_anymore] = [0, 255, 0]  # Green
        plt.imshow(diff_img)
        plt.axis("off")  # Hide axis labels and ticks
        plt.show()

    def show_colonies(self, ax=None):
        if not ax:
            _, ax = plt.subplots(figsize=(8, 16))
        ax.imshow(self.image, cmap="Greys")
        for obj in self.objects:
            obj.add_colonies(ax)
        plt.show()

    def show_summary(self):
        self.show(
            [
                "image",
                "value_thres",
                "closed_value_mask",
                "distance_thres_image",
                "labels",
            ],
            colonies=True,
        )
