from dataclasses import dataclass
from scipy.interpolate import PPoly
from scipy.linalg import det, inv
import numpy as np
from typing import Optional

Point = tuple[float, float]

@dataclass
class LPline:
    """
    Linear piecewise line
    """
    points: list[Point]

    def triangulate(self) -> LPline:
        """
        Treating this LPline as a hull, returns a LPline describing a set of
        triangles filling the hull
        """
        MAX_SEG_LENGTH = 1.0
        if len(LPline.intersection(self, self)) > 0:
            raise ValueError("self intersecting hull")

        # find smallest angle on hole interior
        # connect adjacent or bisect angle with max length line.
        # prefer average angle closest to 60, prefer largest average side length
        hull = self
        while len(hull.points) > 0:
            angles = [hull.angle(i) for i in range(len(hull.points))]
            min_angle = min(angles)
            i = angles.index(min_angle)
            if min_angle <


    def angle(self, i: int):
        _1 = np.array(self.points[i - 1])
        _2 = np.array(self.points[i])
        _3 = np.array(self.points[(i + 1) % len(self.points)])
        return np.acos(np.dot(_1 - _2, _3 - _2) / np.abs(_1 - _2) / np.abs(_3 - _2))

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
