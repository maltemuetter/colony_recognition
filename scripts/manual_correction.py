import sys
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class CorrectSection:
    def __init__(self, section, name):
        self.section = section
        self.name = name
        self.df = None
        self.img = None
        self.fig = None
        self.ax = None
        self.key_handler = None
        self.key_dict = {'e': 'exit', 'n': 'next',
                         'b': 'go back', 'a': 'abort'}

        self.correct_section()

    def is_inside_circle(self, x, y):
        for i, row in self.df.iterrows():
            if np.sqrt((x - row['x']) ** 2 + (y - row['y']) ** 2) < row['r']:
                return True
        return False

    def update_colony_status(self, x, y):
        for i, row in self.df.iterrows():
            if np.sqrt((x - row['x']) ** 2 + (y - row['y']) ** 2) < row['r']:
                if row['identification'] == 'automatic':
                    self.df.at[i, 'activated'] = not row['activated']
                    return
                else:
                    self.df.drop(index=i, inplace=True)
                    return

    def add_manual_colony(self, x, y, new_r=10):
        new_row = pd.DataFrame({'x': [x], 'y': [y], 'r': [new_r], 'activated': [True],
                                'identification': ['manual']})
        self.df = pd.concat([self.df, new_row], ignore_index=True, sort=False)

    def update_image_and_circles(self):
        legend_handles = [mpatches.Patch(color='green', label='Activated'),
                          mpatches.Patch(color='red', label='Deactivated'),
                          mpatches.Patch(color='orange', label='Manual'),
                          mpatches.Patch(color='white', label=''),
                          mpatches.Patch(color='white', label='Keyboard Commands:')]

        for key, value in self.key_dict.items():
            legend_handles.append(mpatches.Patch(
                color='white', label=f'{key}: {value}'))

        self.ax.clear()
        self.ax.imshow(self.img, cmap="gray")
        self.ax.legend(handles=legend_handles, loc="upper left",
                       bbox_to_anchor=(1.02, 0.5))
        self.ax.set_title(self.name)

        for _, row in self.df.iterrows():
            color = 'green' if row['activated'] else 'red'
            if row['identification'] == 'manual':
                color = 'orange'
            self.ax.add_artist(plt.Circle(
                (row['x'], row['y']), row['r'], color=color, fill=False))
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

    def correct_section(self):
        image = self.section.image
        circles = self.section.colonies

        if len(circles) == 0:
            self.df = pd.DataFrame(
                columns=['x', 'y', 'r', 'activated', 'identification'])
        else:
            self.df = pd.DataFrame(circles, columns=['x', 'y', 'r'])

        self.df['activated'] = True
        self.df['identification'] = 'automatic'
        self.img = image

        self.fig, self.ax = plt.subplots(figsize=(10, 20))
        self.update_image_and_circles()

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.show()

        if self.key_handler != "abort":
            self.section.colony_correction_df = self.df
        return self.key_handler
