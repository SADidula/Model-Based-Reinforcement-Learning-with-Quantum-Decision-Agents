import math
from typing import List, Tuple

class PolygonPath:
    """
    Represents a closed polygon path.
    """

    def __init__(self, corner_points: List[Tuple[float, float]]):
        if len(corner_points) < 3:
            raise ValueError("Polygon must have at least 3 vertices!")
        if corner_points[0] != corner_points[-1]:
            corner_points = corner_points + [corner_points[0]]
        self.vertices = corner_points

    def find_closest_segment(self, x: float, y: float):
        """
        Returns (segment_index, (closest_x, closest_y)).
        """
        minimum_distance = float('inf')
        closest_point = None
        index_of_segment = 0

        for i in range(len(self.vertices) - 1):
            ax, ay = self.vertices[i]
            bx, by = self.vertices[i + 1]

            dx = bx - ax
            dy = by - ay
            seg_len2 = dx * dx + dy * dy

            if seg_len2 == 0.0:
                t = 0.0
            else:
                t = ((x - ax) * dx + (y - ay) * dy) / seg_len2
                t = max(0.0, min(1.0, t))
            proj_x = ax + t * dx
            proj_y = ay + t * dy

            distance = math.hypot(x - proj_x, y - proj_y)
            if distance < minimum_distance:
                minimum_distance = distance
                closest_point = (proj_x, proj_y)
                index_of_segment = i

        return index_of_segment, closest_point

    def radial_error(self, x: float, y: float) -> float:
        """
        Signed distance to polygon: positive outside, negative inside.
        """
        _, (cx, cy) = self.find_closest_segment(x, y)
        d = math.hypot(x - cx, y - cy)
        return -d if self.is_point_inside(x, y) else d

    def tangent_heading(self, x: float, y: float) -> float:
        """
        Heading of the closest segment.
        """
        seg_index, _ = self.find_closest_segment(x, y)
        x0, y0 = self.vertices[seg_index]
        x1, y1 = self.vertices[seg_index + 1]
        return math.atan2(y1 - y0, x1 - x0)

    def is_point_inside(self, x: float, y: float) -> bool:
        """
        Ray-casting even-odd rule.
        """
        num_vertices = len(self.vertices) - 1
        inside = False
        tx, ty = x, y
        for i in range(num_vertices):
            x0, y0 = self.vertices[i]
            x1, y1 = self.vertices[i + 1]
            crosses = ((y0 > ty) != (y1 > ty)) and (tx < (x1 - x0) * (ty - y0) / (y1 - y0 + 1e-12) + x0)
            if crosses:
                inside = not inside
        return inside

    def as_points(self) -> List[Tuple[float, float]]:
        return self.vertices

    def centroid(self) -> Tuple[float, float]:
        sum_x = sum([v[0] for v in self.vertices[:-1]])
        sum_y = sum([v[1] for v in self.vertices[:-1]])
        n = len(self.vertices) - 1
        return (sum_x / n, sum_y / n)