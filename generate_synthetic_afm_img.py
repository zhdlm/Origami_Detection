import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt
from skimage.transform import rescale, rotate

CLASS_NAMES = {"Rectangle": 0, "Triangle": 1}

class Config:
    # --- image / physical conventions -------------------------------------- #
    img_size: int = 512
    px_nm: float = 5.0                      # metadata only (1 px = 5 nm)

    # --- dataset composition ----------------------------------------------- #
    negative_frac: float = 0.07             # background-only images (empty labels)
    n_shapes: tuple = (1, 200)              # instances per image (inclusive)
    #touching_image_prob: float = 0.4        # images where touching/overlap allowed

    # --- placement ---------------------------------------------------------- #
    max_overlap: float = 0.35               # max overlapped fraction in "touching" images
    anchor_prob: float = 0.7                # touching images: place next to existing shape

    # --- incompleteness ----------------------------------------------------- #
    broken_prob: float = 0.5                # probability of an object to be broken              
    min_completeness: float = 0.25          # lowest completeness value for an object
    max_broken_completeness: float = 0.97   # lowest completeness value for an object

    # --- heights (nm) ------------------------------------------------------- #
    shape_height_nm: tuple = (1.0, 3.0)     # range of object height
    in_shape_var: float = 0.05              # relative smooth variation on top of shapes


class Shape:

    def __init__(self):
        self.mask = None
        self.C = None

    def scale(self, sx: float, sy: float, img: np.ndarray=None):
        if img is None:
            img = self.mask
        return rescale(img, (sy,sx))

    def rotate(self, theta: float, img: np.ndarray=None):
        if img is None:
            img = self.mask
        return rotate(img, theta, resize=False, center=self.C)

    def translate(self, dx, dy):
        pass

    def remove_parts(self, fraction: float, mode: str):
        pass

    def deform(self, params):
        pass


class AFMScene:

    def __init__(self):

        self.n_instance = random.randint(0, 10)
        self.n_loc = [(random.randint(0,512), random.randint(0,512)) for _ in range(self.n_instance)]
        self.shapes = [random.choice(list(CLASS_NAMES.values())) for _ in range(self.n_instance)]
        self.op = [{"rotate": random.randint(0,360),
                    "scale": random.choice([1, 0.5]),
                    "remove": (random.choice(["Corner", "Lines"]), random.randint(0,100)),
                    "deform": None} for _ in range(self.n_instance)]


    def create_scene(self, shapes):
        scene = np.zeros((512,512), dtype=np.int8)


    def add_background(self, params):
        pass

    def add_topography(self, params):
        pass

    def add_noise(self, params):
        pass

    def add_scan_lines(self):
        pass

    def add_tip_effects(self, params):
        pass

    def render(self):
        pass

    def generate_annotation(self):
        pass

class Rectangle(Shape):

    X = 160 #long side of the rectangle
    Y = 60 #short side of the rectangle
    C = (round(Y/2), round(X/2)) #center of mass (Y,X)

    def create_geometry(self):
        self.mask = np.ones((self.Y,self.X), dtype=bool)
        return self.mask

class Triangle(Shape):

    S_OUT = 126 #Side of the triangle (out)
    S_IN = 47 #Side of the triangle (in)
    H_OUT = round(sqrt(3)*S_OUT/2)
    H_IN = round(sqrt(3)*S_IN/2)
    C = (round(2*H_OUT/3), round(S_OUT/2)) #center of mass (Y,X)

    def __init__(self):

        y, x = np.indices((self.H_OUT, self.S_OUT))
        outer = self.get_triangle_mask(x, y, self.C, self.H_OUT)
        inner = self.get_triangle_mask(x, y, self.C, self.H_IN)
        self.mask = outer & ~inner

    def get_triangle_mask(x: list, y: list, c: tuple, h: int):
        # Centered top vertex and flat base
        y_top = c[0] - (2 / 3) * h
        y_bottom = c[0] + (1 / 3) * h

        # Three bounding conditions: horizontal base + two angled side lines
        base_check = (y >= y_top) & (y <= y_bottom)
        left_edge = y - y_top >= -np.sqrt(3) * (x - c[1])
        right_edge = y - y_top >= np.sqrt(3) * (x - c[1])

        return base_check & left_edge & right_edge

class Smiley(Shape):
    pass

class Ring(Shape):
    pass


img = AFMScene()
print(img.n_instance, img.n_loc, img.shapes, img.op)
# x = [elt[0] for elt in img.n_loc]
# y = [elt[1] for elt in img.n_loc]
# plt.scatter(x,y,c=img.shapes)
# plt.xlim(0,512)
# plt.ylim(0,512)
# plt.show()