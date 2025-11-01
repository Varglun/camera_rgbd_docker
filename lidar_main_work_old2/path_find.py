from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
import cv2 as cv


def find_path(im, start, goal):

    grid = Grid(matrix=(im[:, :, 0] == 0).tolist())  # 0 — проходимо, 1 — препятствие
    start = grid.node(*start)
    end = grid.node(*goal)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, _ = finder.find_path(start, end, grid)

    return path

    # for p in path[1:-1]:
    #     im[p.x, p.y] = (255, 0, 0)
    #     # cv.circle(im, (p.x, p.y), 1, (255, 0, 0), -1)


    # cv.circle(im, (10, 10), 1, (0, 255, 0), -1)
    # cv.circle(im, end_coord, 1, (0, 0, 255), -1)
    # cv.imshow("1", im)
    # cv.waitKey(0)