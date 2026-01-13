from dataclasses import dataclass
from scipy.interpolate import PPoly
from scipy.linalg import det, inv, norm
import numpy as np
from typing import Optional, Any
import unittest
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
from itertools import chain
from matplotlib.animation import ArtistAnimation


Point = tuple[float, float]

@dataclass
class Triangles:
    points: list[tuple[Point, Point, Point]]

    def visualize(self):
        codes = [[Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] for _ in range(len(self.points))]
        vertices = [[p[0], p[1], p[2], p[2]] for p in self.points]
        hull = Path(list(chain(*vertices)), codes=list(chain(*codes)))
        patch = PathPatch(hull, edgecolor="blue", facecolor="none")
        fig, axs = plt.subplots()
        axs.add_patch(patch)
        axs.autoscale_view()
        plt.show()


@dataclass
class LPline:
    """
    Linear piecewise line
    """
    points: list[Point]

    def __getitem__(self, key: int):
        """ Gets items circularly"""
        return self.points[key % len(self.points)]

    def triangulate(self) -> Triangles:
        """
        Treating this LPline as a hull, returns a LPline describing a set of
        triangles filling the hull
        """
        MAX_SEG_LENGTH = 1.0
        if len(LPline.intersection(self, self)) > 0:
            raise ValueError("self intersecting hull")

        # minimize average area? with constraints on angle...?
        #

        # find smallest angle on hole interior
        # connect adjacent or bisect angle with max length line.
        # prefer average angle closest to 60, prefer largest average side length
        # average angle by cbrt(abc)
        # average angle by sqrt(a^2 + b^2 + c^2)
        # average angle by a + b + c + d + ... + <--- including all modified angles
        # connect maximally far vertices
        # not neccesarilty bisect
        # take action which brings angles maximally close to 60, in case of tie prefer largest side lengths
        #   - 120, 180 are equivalently good as 60, allowing later bisection
        #   - smallest average mod 60 (a + b)%c ?= (a%c + b%c)
        #   -
        # always add at least one triangle,
        def construct_triangle(l: Point, r: Point, angle_i: float, angle_j: float) -> Point:
            """ Find a point which constructs a triangle which bisects the given angles
            """
            left = np.array(l)
            right = np.array(r)
            l_hat = (right - left)/norm(right - left)
            lp_hat = np.array([-l_hat[1], l_hat[0]])
            l1 = np.cos(angle_i / 2) * l_hat + np.sin(angle_i / 2) * lp_hat
            l2 = -np.cos(angle_j / 2) * l_hat + np.sin(angle_j / 2) * lp_hat
            b = left - right
            A = np.transpose([-l1, l2]) # check this
            ts = inv(A) @ b
            p = l1 * ts[0] + left
            return p

        hull = self
        fig, axs = plt.subplots()
        artists = []
        covering = Triangles([])
        while len(hull.points) > 3:
            #add frame to animation
            artists.append([hull.as_path()])

            angles = [hull.angle_at(i) for i in range(len(hull.points))]
            min_angle = min(angles)
            i = angles.index(min_angle)
            # TODO: look for triangles several segments away?

            center = np.array(hull[i])
            langle = angles[(i - 1) % len(angles)]
            rangle = angles[(i + 1) % len(angles)]
            # check right cost
            rcost = min_angle / 2 + rangle / 2
            rcost = rcost - rcost * np.floor(rcost / 60.0)

            # check left
            lcost = min_angle / 2 + langle / 2
            lcost = lcost - lcost * np.floor(lcost / 60.0)

            # check connect left - right
            lrcost = LPline.angle((hull[i+1], hull[i-1], hull[i-2])) / 2 + LPline.angle((hull[i+2], hull[i+1], hull[i-1])) / 2
            lrcost = lrcost - lrcost * np.floor(lrcost / 60.0)

            # TODO break ties by maximising either perimeter or area <-- probably area maximising perimeter leads to thin triangles
            # maybe use area instead of angle, area is maximized by equilateral triangle for sides less than some length
            if rcost < lcost and rcost < lrcost:
                new_p = construct_triangle(hull[i], hull[i + 1], min_angle, angles[i + 1])
                proposed = (hull[i], hull[i+1], new_p)
                hull.points.insert(i+1, new_p)
            elif lcost < lrcost:
                new_p = construct_triangle(hull[i], hull[i - 1], angles[i - 1], min_angle)
                proposed = (hull[i-1], hull[i], new_p)
                hull.points.insert(i, new_p)
            else:
                proposed = (hull[i], hull[i+1], hull[i-1])
                hull.points.pop(i)

            covering.points.append(proposed)


        animation = ArtistAnimation(fig=fig, artists=artists)

        plt.show()
        # animation.save("hull.gif")
        return covering


    def angle_at(self, i: int) -> float:
        return LPline.angle((self.points[i - 1], self.points[i], self.points[(i + 1) % len(self.points)]))

    @staticmethod
    def angle(x: tuple[Point, Point, Point]) -> float:
        _1 = np.array(x[0])
        _2 = np.array(x[1])
        _3 = np.array(x[2])
        return np.acos(np.dot(_1 - _2, _3 - _2) / norm(_1 - _2) / norm(_3 - _2))

    @staticmethod
    def area(x: tuple[Point, Point, Point]) -> float:
        _1 = np.array(x[0])
        _2 = np.array(x[1])
        _3 = np.array(x[2])
        return norm(_1 - _2) * np.cross(_1 - _2, _3 - _2) * norm(_3 - _2) / 2

    # divide i -> i + 1 segment n times
    def subdivide(self, i: int, n: int):
        _1 = np.array(self.points[i])
        _2 = np.array(self.points[(i + 1) % len(self.points)])
        def segment(t): return (_2 - _1) * t + _1
        divisions = np.linspace(0, 1, n + 2)
        new_points = [segment(t) for t in divisions[1:-1]]
        self.points = self.points[:i+1] + new_points + self.points[i+1:]

    @staticmethod
    def intersection(a: LPline, b: LPline) -> list[Point]:
        def lline_intersection(a: tuple[Point, Point], b: tuple[Point, Point]) -> Optional[Point]:
            # fast check for bounding box intersection
            axmax, axmin = max([a[0][0], a[1][0]]), min([a[0][0], a[1][0]])
            aymax, aymin = max([a[0][1], a[1][1]]), min([a[0][1], a[1][1]])
            bxmax, bxmin = max([b[0][0], b[1][0]]), min([b[0][0], b[1][0]])
            bymax, bymin = max([b[0][1], b[1][1]]), min([b[0][1], b[1][1]])
            if not (aymax > bymin and aymin < bymax and axmax > bxmin and axmin < bxmax):
                return None
            # calculate intersection
            A = [
                [a[0][0] - a[1][0], -b[0][0] + b[1][0]],
                [a[0][1] - a[1][1], -b[0][1] + b[1][1]],
                ]
            B = [a[1][0] - b[1][0], a[1][1] - b[1][1]]
            x = inv(A) @ b
            if x[0] >= 0 and x[0] <= 1 and x[1] >= 0 and x[1] <= 1:
                return (x[0] * (a[0][0] - a[1][0]) + a[1][0], x[1] * (a[0][1] - a[1][1]) + a[1][1])
            else:
                return None

        intersections = []
        for aa in zip(a.points, a.points[1:] + a.points[0:1]):
            for bb in zip(b.points, b.points[1:] + b.points[0:1]):
                ipoint = lline_intersection(aa, bb)
                if ipoint:
                    intersections.append(ipoint)
        return intersections

    def as_path(self) -> PathPatch:
        codes = [Path.MOVETO] + [Path.LINETO for _ in range(len(self.points) - 2)] + [Path.CLOSEPOLY]
        hull = Path(self.points, codes=codes)
        return PathPatch(hull, edgecolor="blue", facecolor="none")


    def visualize(self, fig_axs=None):
        patch = self.as_path()
        if fig_axs is None:
            fig, axs = plt.subplots()
        else:
            fig, axs = fig_axs
        axs.add_patch(patch)
        axs.autoscale_view()
        return fig, axs


class CoveringTest(unittest.TestCase):
    # @unittest.skip("")
    def test_rectangle(self):
        v0 = np.array((1.0, 0.0))
        v1 = np.array((0.0, 1.0))
        l0 = 10
        l1 = 5
        side1 = [x * v0 for x in np.linspace(0, l0, 5)]
        side2 = [v0 * l0 + x * v1 for x in np.linspace(0, l1, 10)]
        side3 = [(l0 - x) * v0 + v1 * l1 for x in np.linspace(0, l0, 15)]
        side4 = [(l1 - x) * v1 for x in np.linspace(0, l1, 10)]
        hull = LPline(side1 + side2 + side3 + side4)
        # hull.visualize()
        covering = hull.triangulate()
        # covering.visualize()

    @unittest.skip("")
    def test_triangle_visualization(self):
        t = Triangles([((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((2.0, 1.0), (2.0, 0.0), (1.0, 1.0))])
        t.visualize()


if __name__ == "__main__":
    unittest.main()