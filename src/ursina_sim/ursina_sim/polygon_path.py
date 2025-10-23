import math
from typing import List, Tuple

class PolygonPath:
    """
    Represents any closed polygon for path following/obstacle tasks.
    The polygon is defined by a list of at least 3 vertices [(x0, y0), (x1, y1), ...].
    On initialization, it ensures the polygon is closed: the first and last point will always be equal.
    """

    def __init__(self, corner_points: List[Tuple[float, float]]):
        """
        Args:
            corner_points: List of (x, y) coordinate tuples that define the corners of the polygon.
        """
        if len(corner_points) < 3:
            raise ValueError("Polygon must have at least 3 vertices!")
        # Ensure the path closes itself (last vertex = first vertex):
        if corner_points[0] != corner_points[-1]:
            corner_points = corner_points + [corner_points[0]]
        self.vertices = corner_points

    def find_closest_segment(self, x: float, y: float):
        """
        For a given point (x, y), returns:
            - The index of the polygon segment (between two vertices) that is closest.
            - The actual closest point (closest_x, closest_y) on that segment.
        """
        minimum_distance = float('inf')
        closest_point = None
        index_of_segment = 0

        for i in range(len(self.vertices) - 1):
            point_a = self.vertices[i]
            point_b = self.vertices[i + 1]
            ax, ay = point_a
            bx, by = point_b

            # Vector (dx, dy) along the segment
            dx = bx - ax
            dy = by - ay
            segment_length_squared = dx * dx + dy * dy

            # Project the (x, y) onto this segment, clamping to segment ends
            if segment_length_squared == 0:
                t = 0.0
            else:
                # How far along this segment?
                t = ((x - ax) * dx + (y - ay) * dy) / segment_length_squared
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
        Returns the signed shortest distance from point (x, y) to the polygon path.
        Convention:
            - Positive if the point is outside the polygon.
            - Negative if the point is inside.
        """
        _, (closest_x, closest_y) = self.find_closest_segment(x, y)
        distance = math.hypot(x - closest_x, y - closest_y)
        if self.is_point_inside(x, y):
            return -distance
        else:
            return distance

    def tangent_heading(self, x: float, y: float) -> float:
        """
        Returns the angle (in radians) representing the direction of the segment tangent 
        to the polygon at the closest point to (x, y).
        This is 'which way you should be heading' if you want to follow the polygon locally.
        """
        seg_index, _ = self.find_closest_segment(x, y)
        x0, y0 = self.vertices[seg_index]
        x1, y1 = self.vertices[seg_index + 1]
        return math.atan2(y1 - y0, x1 - x0)

    def is_point_inside(self, x: float, y: float) -> bool:
        """
        Determine if a point is inside this polygon.
        Uses the "ray-casting" method (even-odd rule).
        Returns True if (x, y) is inside, False otherwise.
        """
        num_vertices = len(self.vertices) - 1  # (last is duplicate)
        inside = False
        test_x, test_y = x, y
        for i in range(num_vertices):
            x0, y0 = self.vertices[i]
            x1, y1 = self.vertices[i+1]
            crosses = ((y0 > test_y) != (y1 > test_y)) and \
                      (test_x < (x1 - x0) * (test_y - y0) / (y1 - y0 + 1e-12) + x0)
            if crosses:
                inside = not inside
        return inside

    def as_points(self) -> List[Tuple[float, float]]:
        """
        Returns:
            The full list of polygon vertices, including the closing (repeat of the first).
        Useful for visualization or marker plotting.
        """
        return self.vertices

    def centroid(self) -> Tuple[float, float]:
        """
        Attempts to find the centroid (center of mass) of the polygon.
        Simple average for convex polygons.
        """
        sum_x = sum([v[0] for v in self.vertices[:-1]])
        sum_y = sum([v[1] for v in self.vertices[:-1]])
        n = len(self.vertices) - 1  # skip duplicated last
        return (sum_x / n, sum_y / n)