from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import numpy as np


class Obj:
    def __init__(
        self,
        matrix,
        identifier,
        px_min,
        rmin=3,
        overlap_thres=0.75,
        colors={
            "activated": "lime",
            "deactivated": "magenta",
            "manual": "yellow",
        },
    ):
        self.colors = colors
        self.matrix = matrix
        self.overlap_thres = overlap_thres
        self.identifier = identifier
        self.obj = np.where(matrix == identifier, 1, 0)
        self.area = np.sum(self.obj)
        self.use = self.area > px_min
        self.colonies_pre = self.find_colonies(rmin)
        if self.colonies_pre:
            radi = np.array([colony[2] for colony in self.colonies_pre])
            self.rmin = max(max(radi) / 4, rmin)
        else:
            self.rmin = rmin
        self.colonies = self.find_colonies(self.rmin)

    def find_colonies(self, r_min):
        px_min = 3.14 * r_min**2 / 2
        self.radii = ndimage.distance_transform_edt(self.obj)
        radii = self.radii.copy()
        radii[radii < r_min] = 0
        colonies = []
        while True:
            center_y, center_x = np.unravel_index(np.argmax(radii), radii.shape)

            r_best = get_r_best(self.obj, center_x, center_y, r_min=r_min)
            circle = (center_x, center_y, r_best)
            radii = remove_colony(circle, radii, px_min)

            if self.overlap(*circle) > self.overlap_thres:
                colonies.append(circle)

            if np.count_nonzero(radii) == 0:
                return colonies

    def show_colonies(self):
        _, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.obj, cmap="gray")
        self.add_colonies(ax)
        plt.show()

    def add_colonies(self, ax):
        for circle in self.colonies:
            x, y, r = circle
            ax.add_artist(
                plt.Circle(
                    (x, y),
                    r,
                    fill=False,
                    edgecolor=self.colors["activated"],
                )
            )

    def show_obj(self, colonies=False, figsize=(12, 8)):
        _, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.obj, cmap="gray")
        if colonies:
            for circle in self.colonies:
                plt.gca().add_artist(
                    plt.Circle(
                        (circle[0], circle[1]),
                        circle[2],
                        fill=False,
                        edgecolor=self.self.colors["activated"],
                    )
                )
        plt.show()

    def overlap(self, x, y, radius):
        """
        Calculate the overlap between the object area and a circle with given radius and center coordinates.

        Parameters:
            radius (float): The radius of the circle.
            x (int): The x-coordinate of the circle's center.
            y (int): The y-coordinate of the circle's center.

        Returns:
            float: The percentage of overlap between the circle area and the object area.
        """
        circle_mask = self.create_circle_mask(self.matrix.shape, radius, x, y)
        overlap_area = np.sum(np.logical_and(self.obj, circle_mask))
        circle_area = np.pi * radius**2
        overlap = overlap_area / circle_area
        return overlap

    @staticmethod
    def create_circle_mask(shape, radius, x_center, y_center):
        """
        Create a binary mask representing a circle with given radius and center coordinates.

        Parameters:
            shape (tuple): The shape of the mask (height, width).
            radius (float): The radius of the circle.
            x_center (int): The x-coordinate of the circle's center.
            y_center (int): The y-coordinate of the circle's center.

        Returns:
            numpy.ndarray: A binary mask representing the circle.
        """
        y, x = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        mask = dist_from_center <= radius
        return mask


#
#   def circle_obj(self, show: bool = False, circle_thres=0.25):
#       obj = self.obj
#       nonzero_indices = np.nonzero(obj)
#       y_indices = nonzero_indices[0]
#       x_indices = nonzero_indices[1]
#       min_y, max_y, min_x, max_x = (
#           y_indices.min(),
#           y_indices.max(),
#           x_indices.min(),
#           x_indices.max(),
#       )
#
#       if self.area > 4:
#           radius = min(max_x - min_x, max_y - min_y) / 2
#           circleness = np.count_nonzero(obj) / (np.pi * radius**2)
#           self.circleness = circleness
#       else:
#           circleness = 0
#
#       if circleness < circle_thres:
#           self.use = False
#           if show:
#               circle = plt.Circle(
#                   (int(min_x + max_x) / 2, int(min_y + max_y) / 2),
#                   radius,
#                   color="r",
#                   fill=False,
#               )
#               print("\n Object", self.identifier, ": this is probably not a circle.")
#               _, ax = plt.subplots(figsize=(10, 10))
#               ax.imshow(obj)
#               ax.add_artist(circle)
#               plt.show()
#       else:
#           self.use = True
#           if show:
#               circle = plt.Circle(
#                   (int(min_x + max_x) / 2, int(min_y + max_y) / 2),
#                   radius,
#                   color="g",
#                   fill=False,
#               )
#               _, ax = plt.subplots(figsize=(10, 10))
#               ax.imshow(obj)
#               ax.add_artist(circle)
#               plt.show()
#


def get_score(obj, r, x, y, show=False):
    # create a grid of indices corresponding to obj matrix
    X, Y = np.meshgrid(np.arange(obj.shape[1]), np.arange(obj.shape[0]))
    # calculate distance of each grid point to center (x,y)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    # create a mask selecting grid points within radius r
    mask = dist <= r
    # count number of ones and zeros within the masked circle
    count_ones = np.sum(obj[mask])
    count_zeros = np.sum(mask) - count_ones
    # return the score as difference between counts
    if show:
        print("\nzeros: ", count_zeros, "ones: ", count_ones, np.sum(mask))
        plt.imshow(mask)
    return count_ones - 2 * count_zeros


def get_r_best(obj, x, y, r_min=0, show=False):
    score_min = get_score(obj, r_min, x, y)
    score_best = score_min
    r = int(r_min)
    while True:
        r += 1
        score = get_score(obj, r, x, y)
        if score < score_best:
            break
        score_best = score
        if show:
            print(r, score)
    r_best = r - 1
    return r_best


def remove_colony(circle, radii, px_min):
    center_x, center_y, r_best = circle
    X, Y = np.meshgrid(np.arange(radii.shape[1]), np.arange(radii.shape[0]))
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    radii[dist <= r_best] = 0

    labeled, n_objects = ndimage.label(radii)
    for i in range(1, n_objects + 1):
        area = ndimage.sum(labeled == i)
        if area < px_min:
            radii[labeled == i] = 0

    return radii
