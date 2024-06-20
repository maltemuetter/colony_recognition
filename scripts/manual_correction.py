import sys
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from colony_recognition import ColonyRecognition
import matplotlib

# Use the Qt5Agg backend instead of TkAgg
matplotlib.use("Qt5Agg")


class CorrectColonyCount:
    def __init__(
        self,
        recog_obj: ColonyRecognition,
        name,
        colors={
            "activated": "lime",
            "deactivated": "magenta",
            "manual": "yellow",
        },
    ):
        self.recog_obj = recog_obj
        self.name = name
        self.df = None
        self.colors = colors
        self.img = self.recog_obj.image
        self.fig = None
        self.ax = None
        self.key_handler = None
        self.key_dict = {
            "e": "exit",
            "n": "next",
            "b": "go back",
            "a": "abort",
            "u": "unusable",
            "l": "below lower detection limit",
            "u": "above upper detection limit",
        }
        self.auto_correct()

    def is_inside_circle(self, x, y):
        for i, row in self.df.iterrows():
            if np.sqrt((x - row["x"]) ** 2 + (y - row["y"]) ** 2) < row["r"]:
                return True
        return False

    def update_colony_status(self, x, y):
        for i, row in self.df.iterrows():
            if np.sqrt((x - row["x"]) ** 2 + (y - row["y"]) ** 2) < row["r"]:
                if row["identification"] == "automatic":
                    self.df.at[i, "activated"] = not row["activated"]
                    return
                else:
                    self.df.drop(index=i, inplace=True)
                    return

    def add_manual_colony(self, x, y, new_r=10):
        new_row = pd.DataFrame(
            {
                "x": [x],
                "y": [y],
                "r": [new_r],
                "activated": [True],
                "identification": ["manual"],
            }
        )
        self.df = pd.concat([self.df, new_row], ignore_index=True, sort=False)

    def update_image_and_circles(self):
        legend_handles = [
            mpatches.Patch(color=self.colors["activated"], label="Activated"),
            mpatches.Patch(color=self.colors["deactivated"], label="Deactivated"),
            mpatches.Patch(color=self.colors["manual"], label="Manual"),
            mpatches.Patch(color="white", label=""),
            mpatches.Patch(color="white", label="Keyboard Commands:"),
        ]

        for key, value in self.key_dict.items():
            legend_handles.append(
                mpatches.Patch(color="white", label=f"{key}: {value}")
            )

        self.ax.clear()
        self.ax.imshow(self.img, cmap="gray")
        self.ax.legend(
            handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 0.5)
        )
        self.ax.set_title(self.name)

        for _, row in self.df.iterrows():
            color = (
                self.colors["activated"]
                if row["activated"]
                else self.colors["deactivated"]
            )
            if row["identification"] == "manual":
                color = self.colors["manual"]
            self.ax.add_artist(
                plt.Circle((row["x"], row["y"]), row["r"], color=color, fill=False)
            )
        self.fig.canvas.draw()

    def on_press(self, event):
        sys.stdout.flush()
        if event.key in self.key_dict:
            self.key_handler = self.key_dict[event.key]
            plt.close()

    def on_click(self, event):
        sys.stdout.flush()
        x, y = event.xdata, event.ydata
        if self.is_inside_circle(x, y):
            self.update_colony_status(x, y)
        else:
            self.add_manual_colony(x, y)
        self.update_image_and_circles()

    def auto_correct(self):
        circles = self.recog_obj.colonies
        if len(circles) == 0:
            self.df = pd.DataFrame(
                columns=["x", "y", "r", "activated", "identification"]
            )
        else:
            self.df = pd.DataFrame(circles, columns=["x", "y", "r"])
        self.df["activated"] = True
        self.df["identification"] = "automatic"

    def manual_correct(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 20))
        self.update_image_and_circles()

        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.show()

        if self.key_handler != "abort":
            self.recog_obj.colony_correction_df = self.df
        return self.key_handler
