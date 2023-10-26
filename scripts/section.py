import cv2
from scipy import ndimage
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import binary_closing, disk
from .objects import Obj
import warnings


class Section:
    def __init__(self, original_image, background, min_px=200, max_px=5000, d=1, low_thresh=50, d_thresh=5, extra_margin=0.05):
        self.original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        self.image = self.original_image - self.background
        self.image[self.image < 0] = 0
        self.distance_thresholding(
            d=d, low_thresh=low_thresh, d_thresh=d_thresh, extra_margin=extra_margin)
        self.apply_watershed(dist_thresh=3)
        self.size_filter_labels(min_px=min_px, max_px=max_px)
        self.find_objects()
        self.find_colonies()
        self.corrected = False

    def distance_thresholding(self, d=1, low_thresh=50, d_thresh=5, extra_margin=0.05):
        _, thresh = cv2.threshold(
            self.image, low_thresh, 255, cv2.THRESH_BINARY)
        # Define structuring element for the closing operation
        selem = disk(d)
        # Perform morphological closing operation
        closed_thresh_mask = binary_closing(thresh, selem)
        # Get distance to next zero. Thin walls will be very close to zero
        D_closed = ndimage.distance_transform_edt(closed_thresh_mask)
        # Filter out values very close to zero
        _, D_thresh = cv2.threshold(D_closed, d_thresh, 255, cv2.THRESH_BINARY)
        # Create a mask to filter borders
        d = -ndimage.distance_transform_edt(255-D_thresh)
        border_mask = extra_margin-d/np.min(d)
        border_mask[border_mask > 0] = 255
        border_mask[border_mask < 0] = 0
        self.border_mask = border_mask = border_mask.astype(np.uint8)
        # Combine the closed threshold mask with the border mask
        mask = closed_thresh_mask*border_mask
        self.mask = mask.astype(np.int64)
        # find dist to zero again.
        D = ndimage.distance_transform_edt(self.mask)
        self.D = D.astype(np.int64)

    def apply_watershed(self, dist_thresh=3):
        # Apply thresholding to the distance transform
        D_thresh = self.D > dist_thresh

        # Compute the markers using connected components
        self.markers, _ = ndimage.label(D_thresh)

        # Apply the watershed algorithm
        self.labels = watershed(-self.D, self.markers,
                                mask=self.mask, watershed_line=True)

    def size_filter_labels(self, min_px=200, max_px=5000):
        # Remove small objects
        unique, counts = np.unique(self.labels, return_counts=True)
        self.size_filtered_labels = np.zeros_like(self.labels)
        for value, count in zip(unique, counts):
            if (count >= min_px) & (count <= max_px):
                self.size_filtered_labels[self.labels == value] = value

    def show(self, features, figsize=(16, 5)):
        n_features = len(features)
        n_cols = n_features
        _, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        for i, feature in enumerate(features):
            try:
                feature_data = getattr(self, feature)
                if isinstance(feature_data, str):
                    warnings.warn(
                        f"Skipping feature '{feature}' because it's not an image.")
                    continue
                if len(feature_data.shape) == 2:
                    axes[i].imshow(feature_data, cmap='gray')
                else:
                    axes[i].imshow(feature_data)
                axes[i].set_title(feature)
                axes[i].axis('off')
            except AttributeError:
                warnings.warn(f"Skipping unknown feature '{feature}'.")
                continue
        plt.show()

    def find_objects(self):
        self.object_nums = []
        self.objects = []
        for o in np.unique(self.labels):
            obj = Obj(self.labels, o)
            if obj.use:
                self.objects.append(obj)
                self.object_nums.append(o)

    def show_objects(self, n_row=6, figsize=(20, 20)):
        n_objects = len(self.objects)
        n_cols = min(n_objects, n_row)
        n_rows = int(np.floor(n_objects/n_row)+1)

        # Create the figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Iterate through the unique object labels and plot each object in a separate subplot
        for i, obj in enumerate(self.objects):
            # Calculate the subplot indices
            if n_rows == 1:
                idx = i
            else:
                idx = (i // n_cols, i % n_cols)

            # Show the object in the current subplot
            axs[idx].imshow(obj.obj)
            axs[idx].set_title(f"Object {self.object_nums[i]}")
            axs[idx].axis('off')

        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.3, wspace=0.1)

        # Show the figure
        plt.show()

    def find_colonies(self):
        self.colonies = []
        for obj in self.objects:
            obj.find_colonies(r_min=5)
            self.colonies += obj.colonies

    def show_colonies(self, figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.image, cmap="gray")
        for circle in self.colonies:
            plt.gca().add_artist(plt.Circle(
                (circle[0], circle[1]), circle[2], fill=False, edgecolor='green'))
        plt.show()
