import cv2
from scipy import ndimage
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import binary_closing, disk, binary_dilation
from .objects import Obj
import warnings
from icecream import ic


class ColonyRecognition:
    def __init__(
        self,
        d_im: np.ndarray,
        rec_settings={
            "max_pixels": 5000,
            "disk_size": 3,
            "low_threshold": 40,
            "distance_coefficient": 0.6,
            "min_pixels": 15,
            "overlap_threshold": 0.75,
            "min_radius": 3,
        },
    ):
        self.image = d_im
        self.rec_settings = rec_settings
        self.corrected = False
        self.auto_detect_colonies()

    def auto_detect_colonies(self, rec_settings={}):
        settings = self.rec_settings.copy()
        settings.update(rec_settings)

        self.value_thres = self.value_thresholding(
            self.image, settings["low_threshold"]
        )
        self.closed_value_mask = self.fill_holes(
            self.value_thres, settings["disk_size"]
        )
        self.distance_thres_image, self.dist_img = self.distance_thresholding(
            self.closed_value_mask, settings["distance_coefficient"]
        )
        self.labels = self.apply_watershed(self.dist_img, self.distance_thres_image)
        self.size_filtered_labels = self.size_filter_labels(
            self.labels, max_pixels=settings["max_pixels"]
        )
        self.find_objects(
            settings["min_pixels"],
            settings["min_radius"],
            settings["overlap_threshold"],
        )

    @staticmethod
    def value_thresholding(image, low_threshold, show=False):
        _, thresh = cv2.threshold(image, low_threshold, 255, cv2.THRESH_BINARY)
        if show:
            ColonyRecognition.show_difference(image, thresh)
        return thresh

    @staticmethod
    def fill_holes(img, disk_size, show=False):
        selem = disk(disk_size)
        closed_thresh_mask = binary_closing(img, selem)
        if show:
            ColonyRecognition.show_difference(img, closed_thresh_mask)
        return closed_thresh_mask

    @staticmethod
    def distance_thresholding(im_mask, distance_coefficient, show=False):
        D_closed = ndimage.distance_transform_edt(im_mask)
        count, distance = np.histogram(D_closed)
        mean_dist = np.sum(count[2:] * distance[3:]) / np.sum(count[2:])
        d_thres = mean_dist * distance_coefficient
        _, D_thresh = cv2.threshold(D_closed, d_thres, 255, cv2.THRESH_BINARY)
        selem = disk(d_thres)
        restored_edges = binary_dilation(D_thresh, selem)
        restored_image = (
            np.logical_or(restored_edges, D_thresh > 0).astype(np.uint8) * 255
        )
        if show:
            ColonyRecognition.show_difference(im_mask, restored_image)
        return restored_image, ndimage.distance_transform_edt(restored_image)

    @staticmethod
    def apply_watershed(dist_img, dist_thres_image):
        markers, _ = ndimage.label(dist_thres_image)
        labels = watershed(
            -dist_img, markers, mask=dist_thres_image, watershed_line=True
        )
        return labels

    def size_filter_labels(self, labels, max_pixels=5000):
        unique, counts = np.unique(labels, return_counts=True)
        min_pixels = np.median(counts[1:]) / 4
        size_filtered_labels = np.zeros_like(labels)
        for value, count in zip(unique, counts):
            if (count >= min_pixels) & (count <= max_pixels):
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

    def find_objects(self, min_pixels, min_radius, overlap_threshold):
        self.object_nums = []
        self.objects = []
        self.colonies = []
        for o in np.unique(self.labels)[1:]:
            obj = Obj(
                self.labels,
                o,
                min_pixels,
                rmin=min_radius,
                overlap_thres=overlap_threshold,
            )
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
        vmax = min(255, np.median(4 * self.image[self.image > 0]))
        ax.imshow(self.image, cmap="Greys", vmax=vmax)
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
