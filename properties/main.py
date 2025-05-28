import json
import numpy as np
import random
from tqdm import tqdm

base_path = 'C:\\Users\\o.v.naumenko\\Downloads\\arc\\arc-2024\\'


# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


training_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
training_solutions = load_json(base_path + 'arc-agi_training_solutions.json')
evaluation_challenges = load_json(base_path + 'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path + 'arc-agi_evaluation_solutions.json')
test_challenges = load_json(base_path + 'arc-agi_test_challenges.json')
sample_submission = load_json(base_path + 'sample_submission.json')


def process_challenge(challenge_id, challenges,
                      solutions=None):  # solutions=None because test_challenges do not have solutions

    one_challenge = challenges[challenge_id]

    train_inputs = []
    train_outputs = []
    for tasks in one_challenge['train']:
        train_inputs.append(
            np.array(tasks['input']))  # convert to numpy array before you append, so we can see it as a matrix
        train_outputs.append(np.array(tasks['output']))

    test_inputs = []
    for tasks in one_challenge['test']:
        test_inputs.append(np.array(tasks['input']))

    if solutions is not None:
        one_solution = solutions[challenge_id]
        test_outputs = []
        for tasks in one_solution:
            test_outputs.append(np.array(tasks))

        return train_inputs, train_outputs, test_inputs, test_outputs

    else:
        return train_inputs, train_outputs, test_inputs


train_ids = list(training_challenges)

random_id = random.choice(train_ids)
print(random_id)

train_inputs, train_outputs, test_inputs, test_outputs = process_challenge(random_id, training_challenges,
                                                                           training_solutions)


# print([p.shape for p in train_inputs])
# print([p.shape for p in train_outputs])
# print([p.shape for p in test_inputs])
# print([p.shape for p in test_outputs])


# helper functions

# Shapes

# Sets of Pixels should have stats
# Containers should have stats for content
# One task may contain more than one sequence (frame, diagonals, etc.)
# Orientation of "marker" across examples
# sudoku idea - each row and column contains one pixel of each color (check in outputs)
# in corner puzzle, the centerpiece may hint at corner pieces orientation
# mapping may be about mirroring axis
# given several pieces restore others by rotating and mirroring - corner pattern in grid
# coordinates may be relative to fixed points
# check if yellow pixels are symmetric, blue are all the rest
# longest line in bg area (bounding rect coordinates)
# two cells show transformation, third is a test to solve in output
# mask shows colors and priorities
# lines made of objects
# if noise interferes, restore color which should be in that pixel
# rules should be about each shape - do smth or do nothing
# in restore pattern, bg needs to be restored as well
# in dent fill, we may have linear decomposition
# be careful about edges
# 

# ev-168, 163, 120, 85, 198, 200, 258, 355, 398

# Codewords: Constellation (Final solver), planets (individual transformations),
# planets (parameters), meteors (sources of parameters)

def find_objects(matrix):
    height = len(matrix)
    width = len(matrix[0])
    bg = 0
    top_container = []  # consider grouping same shapes, same color shapes identifying for each their unique properties
    top_visited = []
    shape_count = 1

    # each group is a container of its own
    # for bg and non-bg groups locate the other islands inside

    # Locating non-bg hvd groups:
    # go through each pixel, if it is non-bg and not visited, start processing its non-bg neighbors
    # track min/max x/y coordinates (bounding rect) and the number of non-bg pixels (track them by color)
    # if num of non-bg pixels is smaller than num within bounding rect:
    # visit all bounding rect edge pixels and find outer hv bg groups within brect
    # if num of non-bg pixels and outer bg pixels is smaller than num within bounding rect:
    # Recursively: find first internal hv bg groups with non-overlapping brects,
    # find hvd non-bg groups inside those
    # if pixels are not exhausted switch to hvd non-bg groups from prev step and repeat (one by one)
    # now we have a group container with possible islands inside
    # register only internal bg pixels and all non-bg pixels in global visited

    def visit(coord, shape2, neighbors, start_color, same_color, diag):
        # check fit
        i = coord["x"]
        j = coord["y"]

        if starting_color == bg:
            shape2["bg_pixels"].append(coord)
        else:
            shape2["non_bg_pixels"].append(coord)

        if i < shape2["brect"]["min_x"]:
            shape2["brect"]["min_x"] = i
        if j < shape2["brect"]["min_y"]:
            shape2["brect"]["min_y"] = j
        if i > shape2["brect"]["max_x"]:
            shape2["brect"]["max_x"] = i
        if j > shape2["brect"]["max_y"]:
            shape2["brect"]["max_y"] = j
        shape2["colors"][matrix.item((i, j))] += 1

        if diag:
            neis = [(i - 1, j), (i + 1, j), (i - 1, j - 1), (i + 1, j + 1), (i - 1, j + 1), (i + 1, j - 1), (i, j - 1),
                    (i, j + 1)]
            for nei in neis:
                if 0 <= nei[0] < height and 0 <= nei[1] < width:
                    neighbor = {"x": nei[0], "y": nei[1], "color": matrix.item((nei[0], nei[1]))}
                    color_check = True
                    if starting_color == bg:
                        color_check = neighbor["color"] == bg
                    else:
                        color_check = neighbor["color"] > bg
                    if neighbor not in neighbors and neighbor not in top_visited and color_check:
                        neighbors.append(neighbor)
        else:
            neis = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for nei in neis:
                if 0 <= nei[0] < height and 0 <= nei[1] < width:
                    neighbor = {"x": nei[0], "y": nei[1], "color": matrix.item((nei[0], nei[1]))}
                    color_check = True
                    if starting_color == bg:
                        color_check = neighbor["color"] == bg
                    else:
                        color_check = neighbor["color"] > bg
                    if neighbor not in neighbors and neighbor not in top_visited and color_check:
                        neighbors.append(neighbor)

        for neighbor in neighbors:
            if neighbor in top_visited:
                continue
            top_visited.append(neighbor)
            visit(neighbor, shape2, neighbors, start_color, same_color, diag)

    def analyze_shapes(shape2):
        # shape
        shape2["shape"] = np.array(matrix[shape2["brect"]["min_x"]:shape2["brect"]["max_x"] + 1,
                                   shape2["brect"]["min_y"]:shape2["brect"]["max_y"] + 1], copy=True)
        # clean it up
        if shape2["non_bg_pixels"]:
            for i in range(shape2["brect"]["min_x"], shape2["brect"]["max_x"] + 1):
                for j in range(shape2["brect"]["min_y"], shape2["brect"]["max_y"] + 1):
                    if matrix.item((i, j)) > 0:
                        tmp = {"x": i, "y": j, "color": matrix.item((i, j))}
                        if tmp not in shape2["non_bg_pixels"]:
                            shape2["shape"][i - shape2["brect"]["min_x"]][j - shape2["brect"]["min_y"]] = 0
        else:
            for i in range(shape2["brect"]["min_x"], shape2["brect"]["max_x"] + 1):
                for j in range(shape2["brect"]["min_y"], shape2["brect"]["max_y"] + 1):
                    if matrix.item((i, j)) == 0:
                        tmp = {"x": i, "y": j, "color": 0}
                        if tmp not in shape2["bg_pixels"]:
                            shape2["shape"][i - shape2["brect"]["min_x"]][j - shape2["brect"]["min_y"]] = 1

        # islands
        # for each colored group find the smallest bounding bg group
        # for each bg group find the smallest bounding colored group
        # only bigger brects count, touching_edge belongs in top_container
        parent = {"brect": {"min_x": -1, "min_y": -1, "max_x": height, "max_y": width}, "islands": []}
        for shape in top_container:
            if shape2["brect"]["max_y"] < shape["brect"]["max_y"] < parent["brect"]["max_y"] and shape2["brect"][
                "max_x"] < shape["brect"]["max_x"] < parent["brect"]["max_x"] and shape2["brect"]["min_y"] > \
                    shape["brect"]["min_y"] > parent["brect"]["min_y"] and shape2["brect"]["min_x"] > shape["brect"][
                "min_x"] > parent["brect"]["min_x"]:
                # check opposite color
                parent = shape
        parent["islands"].append(shape2)

        # contour, touching_edge
        outer_bg = []
        # if any coordinate of brect is 0 or height-1/width-1 then touching_edge
        if shape2["brect"]["min_x"] == 0 or shape2["brect"]["min_y"] == 0 or shape2["brect"]["max_x"] == height - 1 or \
                shape2["brect"]["max_y"] == width - 1:
            shape2["touching_edge"] = True

        # visit all bounding rect edge pixels and find outer hv bg groups within brect
        for coord in range(shape2["brect"]["max_y"]):
            pixel = {"x": shape2["brect"]["min_x"], "y": coord, "color": matrix.item((shape2["brect"]["min_x"], coord))}
            if pixel["color"] == bg and pixel not in outer_bg:
                outer_bg.append(pixel)
            if pixel["color"] != bg and pixel not in shape2["contour"]:
                shape2["contour"].append(pixel)
        for coord in range(shape2["brect"]["max_x"]):
            pixel = {"x": coord, "y": shape2["brect"]["min_y"], "color": matrix.item((coord, shape2["brect"]["min_y"]))}
            if pixel["color"] == bg and pixel not in outer_bg:
                outer_bg.append(pixel)
            if pixel["color"] != bg and pixel not in shape2["contour"]:
                shape2["contour"].append(pixel)
        for coord in range(shape2["brect"]["max_y"]):
            pixel = {"x": shape2["brect"]["max_x"] - 1, "y": coord,
                     "color": matrix.item((shape2["brect"]["max_x"] - 1, coord))}
            if pixel["color"] == bg and pixel not in outer_bg:
                outer_bg.append(pixel)
            if pixel["color"] != bg and pixel not in shape2["contour"]:
                shape2["contour"].append(pixel)
        for coord in range(shape2["brect"]["max_x"]):
            pixel = {"x": coord, "y": shape2["brect"]["max_y"] - 1,
                     "color": matrix.item((coord, shape2["brect"]["max_y"] - 1))}
            if pixel["color"] == bg and pixel not in outer_bg:
                outer_bg.append(pixel)
            if pixel["color"] != bg and pixel not in shape2["contour"]:
                shape2["contour"].append(pixel)

        def visit_bg(pixel3, shape2):
            i = pixel3["x"]
            j = pixel3["y"]
            neis = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for nei in neis:
                if shape2["brect"]["min_x"] <= nei[0] <= shape2["brect"]["max_x"] and shape2["brect"]["min_y"] <= nei[
                    1] <= shape2["brect"]["max_y"] and matrix.item((nei[0], nei[1])) == 0:
                    nei_bg = {"x": nei[0], "y": nei[1], "color": matrix.item((nei[0], nei[1]))}
                    if nei_bg not in outer_bg and nei_bg not in visited_bg:
                        outer_bg.append(nei_bg)

        # visit hv bg neighbors within brect
        visited_bg = []

        for bg_neighbor in outer_bg:
            if bg_neighbor in visited_bg:
                continue
            visited_bg.append(bg_neighbor)
            visit_bg(bg_neighbor, shape2)

        # all shape pixels with neighbors from outer_bg belong to contour
        for pix in shape2["non_bg_pixels"]:
            contour_check = False
            i_p = pix["x"]
            j_p = pix["y"]
            neis = [(i_p - 1, j_p), (i_p + 1, j_p), (i_p, j_p - 1), (i_p, j_p + 1)]
            for nei in neis:
                tmp_bg = {"x": nei[0], "y": nei[1], "color": 0}
                if tmp_bg in outer_bg:
                    contour_check = True
                    break
            if contour_check and pix not in shape2["contour"]:
                shape2["contour"].append(pix)

        # contour pixels with contour neighbors with one coordinate equal belong to edge
        # find edges (their number may be useful)

        # contour pixels with contour neighbors with no equal coordinates are vertices
        # find vertices (their properties may be useful)

        # inside
        # shape pixels that are not in contour are "inside"
        for pix in shape2["non_bg_pixels"]:
            if pix not in shape2["contour"]:
                shape2["inside"].append(pix)

        # neighbors
        # include neighbors of contour pixels from outer_bg or outside brect if available
        for pix in shape2["contour"]:
            i_p = pix["x"]
            j_p = pix["y"]
            neis = [(i_p - 1, j_p), (i_p + 1, j_p), (i_p - 1, j_p - 1), (i_p + 1, j_p + 1), (i_p - 1, j_p + 1),
                    (i_p + 1, j_p - 1), (i_p, j_p - 1),
                    (i_p, j_p + 1)]
            for nei in neis:
                if shape2["brect"]["min_x"] <= nei[0] <= shape2["brect"]["max_x"] and shape2["brect"]["min_y"] <= nei[
                    1] <= shape2["brect"]["max_y"]:
                    if matrix.item((nei[0], nei[1])) == 0:
                        tmp_bg = {"x": nei[0], "y": nei[1], "color": 0}
                        if tmp_bg in outer_bg and tmp_bg not in shape2["neighbors"]:
                            shape2["neighbors"].append(tmp_bg)
                if shape2["brect"]["min_x"] > nei[0] or nei[0] > shape2["brect"]["max_x"] or shape2["brect"]["min_y"] > \
                        nei[1] or nei[1] > shape2["brect"]["max_y"]:
                    if 0 <= nei[0] < height and 0 <= nei[1] < width:
                        tmp_bg = {"x": nei[0], "y": nei[1], "color": 0}
                        if tmp_bg not in shape2["neighbors"]:
                            shape2["neighbors"].append(tmp_bg)

        def visit_color(pixel3, shape2, st_color, diag):
            i = pixel3["x"]
            j = pixel3["y"]

            if i < shape2["brect"]["min_x"]:
                shape2["brect"]["min_x"] = i
            if j < shape2["brect"]["min_y"]:
                shape2["brect"]["min_y"] = j
            if i > shape2["brect"]["max_x"]:
                shape2["brect"]["max_x"] = i
            if j > shape2["brect"]["max_y"]:
                shape2["brect"]["max_y"] = j

            if diag:
                neis = [(i_p - 1, j_p), (i_p + 1, j_p), (i_p - 1, j_p - 1), (i_p + 1, j_p + 1), (i_p - 1, j_p + 1),
                        (i_p + 1, j_p - 1), (i_p, j_p - 1),
                        (i_p, j_p + 1)]
            else:
                neis = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

            for nei in neis:
                if shape2["brect"]["min_x"] <= nei[0] <= shape2["brect"]["max_x"] and shape2["brect"]["min_y"] <= nei[
                    1] <= shape2["brect"]["max_y"] and matrix.item((nei[0], nei[1])) == st_color:
                    nei_color = {"x": nei[0], "y": nei[1], "color": st_color}
                    if nei_color in color_shape["non_bg_pixels"] and nei_color not in visited_non_bg:
                        color_neighbors.append(nei_color)

            for neighbor in color_neighbors:
                if neighbor in visited_non_bg:
                    continue
                visited_non_bg.append(neighbor)
                visit_color(neighbor, shape2, st_color, diag)

        # find same color hv shapes
        visited_non_bg = []
        color_neighbors = []

        for pix in shape2["non_bg_pixels"]:
            if pix in visited_non_bg:
                continue
            color_shape = {"brect": {}, "color": pix["color"], "non_bg_pixels": [], "shape": []}
            color_shape["brect"]["min_x"] = height
            color_shape["brect"]["min_y"] = width
            color_shape["brect"]["max_x"] = 0
            color_shape["brect"]["max_y"] = 0
            color_shape["non_bg_pixels"].append(pix)
            visited_non_bg.append(pix)
            visit_color(pix, shape2, pix["color"], False)
            shape2["shape"] = np.array(matrix[shape2["brect"]["min_x"]:shape2["brect"]["max_x"] + 1,
                                       shape2["brect"]["min_y"]:shape2["brect"]["max_y"] + 1], copy=True)
            for i in range(shape2["brect"]["min_x"], shape2["brect"]["max_x"] + 1):
                for j in range(shape2["brect"]["min_y"], shape2["brect"]["max_y"] + 1):
                    if matrix.item((i, j)) > 0:
                        tmp = {"x": i, "y": j, "color": matrix.item((i, j))}
                        if tmp not in shape2["non_bg_pixels"]:
                            shape2["shape"][i - shape2["brect"]["min_x"]][j - shape2["brect"]["min_y"]] = 0
            shape2["same_color_hv"].append(color_shape)

        # find same color hvd shapes
        visited_non_bg = []
        color_neighbors = []

        for pix in shape2["non_bg_pixels"]:
            if pix in visited_non_bg:
                continue
            color_shape = {"brect": {}, "color": pix["color"], "non_bg_pixels": [], "shape": []}
            color_shape["brect"]["min_x"] = height
            color_shape["brect"]["min_y"] = width
            color_shape["brect"]["max_x"] = 0
            color_shape["brect"]["max_y"] = 0
            color_shape["non_bg_pixels"].append(pix)
            visited_non_bg.append(pix)
            visit_color(pix, shape2, pix["color"], True)
            shape2["shape"] = np.array(matrix[shape2["brect"]["min_x"]:shape2["brect"]["max_x"] + 1,
                                       shape2["brect"]["min_y"]:shape2["brect"]["max_y"] + 1], copy=True)
            for i in range(shape2["brect"]["min_x"], shape2["brect"]["max_x"] + 1):
                for j in range(shape2["brect"]["min_y"], shape2["brect"]["max_y"] + 1):
                    if matrix.item((i, j)) > 0:
                        tmp = {"x": i, "y": j, "color": matrix.item((i, j))}
                        if tmp not in shape2["non_bg_pixels"]:
                            shape2["shape"][i - shape2["brect"]["min_x"]][j - shape2["brect"]["min_y"]] = 0
            shape2["same_color_hvd"].append(color_shape)

        # check if mono-color and if so which it is

        # is_symmetric
        res, res2 = is_symmetric(shape2["shape"])
        if res != "None" and res2 != "None":
            shape2["is_symmetric"] = True

    def generalize_shape(shape):
        height2 = len(shape)
        width2 = len(shape[0])
        gen_shape = np.copy(shape["shape"])
        finished = True
        while finished:
            if height2 < 2 and width2 < 2:
                break
            i1 = 0
            while width2 > 1 and i1 < width2 - 1:
                i2 = i1 + 1
                if np.array_equal(gen_shape[:, i1], gen_shape[:, i2]):
                    gen_shape = np.delete(gen_shape, i2, 1)
                    width2 = len(gen_shape[0])
                else:
                    i1 += 1
            i1 = 0
            while height2 > 1 and i1 < height2 - 1:
                i2 = i1 + 1
                if np.array_equal(gen_shape[i1, :], gen_shape[i2, :]):
                    gen_shape = np.delete(gen_shape, i2, 0)
                    height2 = len(gen_shape)
                else:
                    i1 += 1
        return gen_shape

    # start at 0,0 and find all neighbors meeting constraints
    neighbors = []
    for ii in range(height):
        for jj in range(width):
            curr = {"x": ii, "y": jj, "color": matrix.item((ii, jj))}
            if curr not in top_visited:
                shape = {"brect": {}, "colors": {}, "non_bg_pixels": [], "bg_pixels": [], "contour": [], "edges": [],
                         "vertexes": [], "inside": [], "neighbors": [], "touching_edge": False, "is_symmetric": False,
                         "islands": [], "shape": [], "generalized_shape": [], "same_color_hv": [], "same_color_hvd": []}
                shape["brect"]["min_x"] = height
                shape["brect"]["min_y"] = width
                shape["brect"]["max_x"] = 0
                shape["brect"]["max_y"] = 0
                for c in range(10):
                    shape["colors"][c] = 0
                starting_color = matrix.item((ii, jj))
                top_visited.append(curr)
                neighbors.append(curr)
                if starting_color == bg:
                    visit(curr, shape, neighbors, starting_color, True, False)
                else:
                    visit(curr, shape, neighbors, starting_color, False, True)
                top_container.append(shape)
                shape_count += 1

    for shape in top_container:
        analyze_shapes(shape)

    return top_container

    # for each non-bg group in the process figure out:
    # - total set of non-bg pixels - others are outside
    # - edge pixels (corner pixels and edges pixels, num of edge lines)
    # - inside pixels (stats - number of pixels, number of colors, bg holes)
    # - immediate outside neighbor pixels (including outer ones - touching_edge)
    # - one-color hv groups (rectangle sections within them)
    # - one color hvd groups
    # - bounding rect, its coordinates, pixel coordinates are relative to its top left corner
    # - all other pixels in bounding rect are "assumed" bg (it may be non-zero bg)
    # - quadrants, is_symmetric

    # Special cases (if needed, lazy evaluation):
    # - dents and protrusions (their dimensions may be determined dynamically)
    # - shape representation ? (generalization?)
    # - prepare to number pixels from left to right and from top to bottom starting from top left or other order
    # - if one gap in edge - consider "imaginary inside area" (check bounding rect edges)

    # check if group is a line or rectangle (square, rhombus, trapezoid, parallelepiped)
    # for a line check gaps, halves of array, continuations, end and midpoints, any sequence
    # for polygons try to figure out type

    # locate bg hv groups


def check_objects_line(shape):
    # lines made of objects may keep object symmetry or rotation through transformation
    pass


def check_line(shape):
    # sort pixels by coordinates - should be 1 pixel per coordinate
    # check slope - should be consistent
    # check gaps
    # fix endpoints and midpoint, dx, dy, length
    # neighbors
    # missing points to split screen - its halves
    # with thick lines be careful about edges
    pass


def distance_to_line(pixel, line):
    # draw perpendiculars by manipulating line's slope
    # calculate number of pixels to reach line
    pass


def check_polygon(shape):
    # check number of edges and vertices
    # check slopes of edges - problems with non-hv edges (not 1 pixel width)
    # diagonal, center lines
    # quadrants
    pass


def process_cell(matrix):
    # non-connectedness is not a problem
    # array, color stats, same color groups, is_symmetric
    pass


def find_shapes(matrix, same_color, diag):
    height = len(matrix)
    width = len(matrix[0])
    shapes = []
    visited = []

    def visit(coord, shape, start_color):
        # check fit
        i = coord[0]
        j = coord[1]
        part1 = 0 <= i < height and 0 <= j < width
        if not part1:
            return
        part2 = matrix.item((i, j)) == start_color
        part3 = start_color == 0 and matrix.item((i, j)) == 0
        part4 = start_color > 0 and matrix.item((i, j)) > 0
        if part1 and (part2 or (not same_color and (part3 or part4))) and (i, j) not in visited:
            visited.append((i, j))
            shape.append((i, j))
            neighbors = []
            if diag:
                neighbors = [(i - 1, j), (i + 1, j), (i - 1, j - 1), (i + 1, j + 1), (i - 1, j + 1), (i + 1, j - 1),
                             (i, j - 1), (i, j + 1)]
            else:
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for neighbor in neighbors:
                visit(neighbor, shape, start_color)

    # start at 0,0 and find all neighbors meeting constraints
    for i in range(height):
        for j in range(width):
            if not (i, j) in visited:
                shape = []
                start_color = matrix.item((i, j))
                visit((i, j), shape, start_color)
                shapes.append(shape)
    return shapes


def calculate_stats(shapes):
    # group shapes - find rarest/most frequent

    # group colors

    # calculate number of pixels

    # neighbor stats

    # container stats

    # coordinate properties - leftmost
    pass


def copy_shape(shape1, output, bounding_rect):
    pass


def fill_output(output, coordinates, colors):
    pass


def prepare_output(h, w, bg):
    pass


# Compare
# multicolor hvd shape or one-color hv part of shape
# grid "coordinates" - minmodel idea
# corner pattern - use 2 attempts for submission to check different orientations + respect symmetry

def compare_shapes(shape_container_in, shape_container_out):
    results = {}
    # 1. Dimensions, number of groups, color stats - collect above
    # 2. For each input shape: check if each color subgroup is in output
    # (as is, scaled/minmodeled, mirrored(if symmetric - half or quarter, copies), rotated,
    # inverted, colored; check for by line or overlapping or generalization;
    # check against one-color groups or subgroups (total inclusion in bigger group counts).
    # First check dimensions and color stats - generalization, edge overflow, rotated rectangle
    # and decomposition allow for different dimensions, edge overflow and overlapping allow for
    # different color stats, otherwise expect equality
    #
    # always check for generalization (if any 2 lines are equal remove 1 of them, h and v)
    # if sh1 < sh2: check for copies, inclusion, decomposition, scaling/minmodeling and tiling
    # if sh1 == sh2: check for equality, symmetry, rotation, inversion
    # if sh1 > sh2: check for reverse "copies, inclusion, decomposition, scaling/minmodeling and tiling"

    # check position of pattern from all possible sides even if the first check is successful - options

    in_checked = []
    out_checked = []
    match_found = False
    for sh in shape_container_in:
        for sh2 in sh["same_color_hv"]:
            for sh_out in shape_container_in:
                for sh_out2 in sh_out["same_color_hv"]:
                    h1 = len(sh2["shape"])
                    w1 = len(sh2["shape"][0])
                    h2 = len(sh_out2["shape"])
                    w2 = len(sh_out2["shape"][0])
                    # check for inclusion (overlapping), decomposition, scaling/minmodeling
                    if h1 > h2 and w1 > w2:
                        pass
                    elif h1 > h2:
                        pass
                    elif w1 > w2:
                        pass
                    # check for equality, symmetry, rotation, inversion
                    elif h1 == h2 and w1 == w2:
                        pass
                    # check for inclusion, decomposition, scaling/minmodeling
                    elif h1 < h2 and w1 < w2:
                        pass
                    elif h1 < h2:
                        pass
                    elif w1 < w2:
                        pass
        for sh2 in sh["same_color_hvd"]:
            for sh_out in shape_container_in:
                for sh_out2 in sh["same_color_hvd"]:
                    pass
        for sh2 in sh["islands"]:
            for sh_out in shape_container_in:
                for sh_out2 in sh["islands"]:
                    pass

    # 3. For each output shape, how does it appear in input; see p. 2
    in_checked = []
    out_checked = []
    for sh in shape_container_out:
        pass

    # 4. Compare input shapes (task ev-53 - compare 2 shapes in input, place colored diff in output - new transform),
    # group shapes by shape (generalized), color, size
    in_checked = []
    for sh in shape_container_in:
        pass

    # check for noise in output - if absent, consider stats by lines to employ majority vote

    # match, partial_match (noise, overlap, edge, mismatching pieces), no_match

    # avoid double use!

    return results


def compare_decomposed_shapes(shape1, shape2):
    # given a shape in some color and
    # a set of shapes in that color or mapped with the same number of pixels
    # compare by lines, expect whole shapes from set to fit
    # keep track of dx, dy for each shape by some index in the whole shape
    pass


# Color Inversion

def check_inversion(shape1, shape2):
    # check they have only 2 colors
    pass


def invert_colors(shape):
    # check it has only 2 colors
    pass


# Symmetry
# prepare a function to find symmetry faults

def is_symmetric(shape):
    # rect or square
    flags = [True, True]
    # check if top left is symmetric with top right and bottom left
    height = len(shape)
    width = len(shape[0])
    for i in range(height // 2):
        for j in range(width // 2):
            val = shape[i, j]
            if flags[0]:
                flags[0] = val == shape[i, width - j - 1]
            if flags[1]:
                flags[1] = val == shape[height - i - 1, j]
            if not flags[0] and not flags[1]:
                break
    # check if bottom right is symmetric with top right and bottom left
    for i in range(height // 2):
        for j in range(width // 2):
            val = shape[height - i - 1, width - j - 1]
            if flags[1]:
                flags[1] = val == shape[i, width - j - 1]
            if flags[0]:
                flags[0] = val == shape[height - i - 1, j]
            if not flags[0] and not flags[1]:
                break
    # do not return for now
    result = "None"
    if flags[0] and flags[1]:
        result = "both"
    elif flags[0]:
        result = "vertical"
    elif flags[1]:
        result = "horizontal"

    result2 = "None"
    flags2 = [True, True]
    if height == width:  # only square
        # check if left diagonal quarter is symmetric to top and bottom diagonal quarters
        for i in range(height):
            for j in range(min(i, height - i)):
                val = shape[i, j]
                if flags2[0]:
                    flags2[0] = val == shape[j, i]
                if flags2[1]:
                    flags2[1] = val == shape[j, height - i]
                if not flags2[0] and not flags2[1]:
                    break
        # check if right diagonal quarter is symmetric to top and bottom diagonal quarters
        for i in range(height):
            for j in range(min(i, height - i)):
                val = shape[i, j]
                if flags2[0]:
                    flags2[0] = val == shape[j, i]
                if flags2[1]:
                    flags2[1] = val == shape[j, height - i]
                if not flags2[0] and not flags2[1]:
                    break
        # do not return for now
        if flags2[0] and flags2[1]:
            result2 = "both"
        elif flags2[0]:
            result2 = "top left"
        elif flags2[1]:
            result2 = "bottom left"

    if flags[0] and flags[1] and flags2[0] and flags2[1]:
        result2 = "central"

    return result, result2


def is_symmetric_to(shape, shape2):
    # rect or square - check for h or v symmetry
    flags = [True, True]
    height = len(shape)
    width = len(shape[0])
    for i in range(height):
        for j in range(width):
            val = shape[i, j]
            if flags[0]:
                flags[0] = val == shape2[i, width - j - 1]
            if flags[1]:
                flags[1] = val == shape[height - i - 1, j]
            if not flags[0] and not flags[1]:
                break
    # square only - check for d1 or d2 symmetry
    flags2 = [True, True]
    for i in range(height):
        for j in range(width):
            val = shape[i, j]
            if flags2[0]:
                flags2[0] = val == shape2[j, i]
            if flags2[1]:
                flags2[1] = val == shape[width - j - 1, height - i - 1]
            if not flags2[0] and not flags2[1]:
                break
    if flags[0] and flags[1]:
        result = "hv"
    elif flags[0]:
        result = "h"
    elif flags[1]:
        result = "v"
    result = 0
    result2 = 0
    if flags2[0] and flags2[1]:
        result2 = "d1d2"
    elif flags2[0]:
        result2 = "d1"
    elif flags2[1]:
        result2 = "d2"
    return result, result2


def mirror(shape, axis):
    pass


def point_mirror(shape, pixel):
    pass


def object_mirror(shape, axis, objects):
    # objects should not mirror, only their positions mirror
    pass


def matrix_flip(m, d):
    myl = np.array(m)
    if d == 'v':
        return np.flip(myl, axis=0)
    elif d == 'h':
        return np.flip(myl, axis=1)


# diagonal flip
def matrix_flip_diag(a):
    a.T = np.rot90(np.fliplr(a))


# Rotation

def is_rotation(shape, shape2):
    flags = [True, True, True]
    height = len(shape)
    width = len(shape[0])
    height2 = len(shape2)
    width2 = len(shape2[0])
    total = width * height
    current = 0
    while current < total:
        val = shape(current // height, current % width)
        if flags[0] and height == width2 and width == height2:
            flags[0] = val == shape2(current % width, height - current // height - 1)
        if flags[1]:
            flags[1] = val == shape2(height - current // height - 1, width - current % width - 1)
        if flags[2] and height == width2 and width == height2:
            flags[2] = val == shape2(width - current % width - 1, current // height)
        if not flags[0] and not flags[1] and not flags[2]:
            break
        current = current + 1
    if flags[0] and flags[1] and flags[2]:
        return "symmetric"
    elif flags[0]:
        return 90
    elif flags[1]:
        return 180
    elif flags[2]:
        return 270
    return False


def rotate(shape, angle):
    pass


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html

# rotated = numpy.rot90(orignumpyarray,3) # 270 degrees

# Function to rotate matrix 90 degree clockwise
def rotate90_clockwise(A):
    N = len(A[0])
    for i in range(N // 2):
        for j in range(i, N - i - 1):
            temp = A[i][j]
            A[i][j] = A[N - 1 - j][i]
            A[N - 1 - j][i] = A[N - 1 - i][N - 1 - j]
            A[N - 1 - i][N - 1 - j] = A[j][N - 1 - i]
            A[j][N - 1 - i] = temp


# Rotate Matrix by 180 degrees
def rotate_matrix2(data):
    rows = len(data)
    cols = len(data[0])
    # Reversing all rows
    for i in range(len(data)):
        data[i] = data[i][::-1]
    # Reversing all rows of the matrix
    data = data[::-1]


# Function to rotate a matrix by one place
def rotate_matrix(mat):
    if not len(mat):
        return
    top = 0
    bottom = len(mat) - 1

    left = 0
    right = len(mat[0]) - 1

    while left < right and top < bottom:

        # Store the first element of next row,
        # this element will replace first element of
        # current row
        prev = mat[top + 1][left]

        # Move elements of top row one step right
        for i in range(left, right + 1):
            curr = mat[top][i]
            mat[top][i] = prev
            prev = curr

        top += 1

        # Move elements of rightmost column one step downwards
        for i in range(top, bottom + 1):
            curr = mat[i][right]
            mat[i][right] = prev
            prev = curr

        right -= 1

        # Move elements of bottom row one step left
        for i in range(right, left - 1, -1):
            curr = mat[bottom][i]
            mat[bottom][i] = prev
            prev = curr

        bottom -= 1

        # Move elements of leftmost column one step upwards
        for i in range(bottom, top - 1, -1):
            curr = mat[i][left]
            mat[i][left] = prev
            prev = curr

        left += 1

    return mat


# Scale

def get_ratios(dim_x, dim_y):
    ratios = []
    for i in range(1, min(dim_x, dim_y) + 1):
        if dim_x % i == dim_y % i == 0:
            ratios.append(i)
    ratios.remove(1)
    return ratios


def check_ratio(shape, ratio):
    # check that each square ratio is of the same color
    # update for occluded or noisy shape
    dim_x, dim_y = shape.shape()

    return True


def scale_down(shape):
    height = len(shape)
    width = len(shape[0])
    ratios = get_ratios(height, width)
    for r in ratios:
        pass
    # go through the 1st row, find and filter ratios
    # go down each column, filter ratios
    # ignore if shape is touching edge and the finishing pixels are missing


def is_scaled(shape):
    dim_x, dim_y = shape.shape()  # ???

    ratios = get_ratios(dim_x, dim_y)  # get possible ratios bigger than 1

    for r in ratios:
        if not check_ratio(shape, r):
            ratios.remove(r)

    if ratios:
        return max(ratios)

    return 1


def scale(shape, ratio):
    pass


# Repetitions

def find_repetitions(line):
    rep = str(line[0])
    index = 1
    while index < len(line):
        if line[index] != rep[0]:
            rep = rep + line[index]
            index = index + 1
        else:
            checked = True
            index2 = index + 1
            count = 1
            while index2 < len(line):
                while index2 < len(rep) * count and line[index2] == line[index2 - len(rep) * count]:
                    index2 = index2 + 1
                if index2 == len(rep) * count:
                    count = count + 1
                if line[index2] != line[index2 - len(rep) * count]:
                    checked = False
                    break
            if checked:
                return True
            else:
                rep = rep + line[index]
                index = index + 1

    if len(rep) < len(line):
        return rep
    else:
        return False


def continue_seq(rep):
    pass


def general_color_stats(shapes, matrix):
    obj = {"colors": {}}
    for c in range(10):
        obj["colors"][c] = {"pixels": 0, "shapes": 0}
    height = len(matrix)
    width = len(matrix[0])
    for i in range(height):
        for j in range(width):
            obj["colors"][matrix.item((i, j))]["pixels"] += 1
    for sh in shapes:
        color = -1
        for i in range(10):
            if sh["colors"][i] > 0:
                if color == -1:
                    color = i
                else:
                    color = 11
        if color != 11:
            obj["colors"][color]["shapes"] += 1
    return obj


def check_dimensions(input_shape, output_shape):
    obj = {"input": {"height": len(input_shape), "width": len(input_shape[0])},
           "output": {"height": len(output_shape), "width": len(output_shape[0])},
           "diff": {"height": "equal", "width": "equal"}}
    if obj["input"]["height"] > obj["output"]["height"]:
        obj["diff"]["height"] = "input_bigger"
    elif obj["input"]["height"] < obj["output"]["height"]:
        obj["diff"]["height"] = "input_smaller"
    if obj["input"]["width"] > obj["output"]["width"]:
        obj["diff"]["width"] = "input_bigger"
    elif obj["input"]["width"] < obj["output"]["width"]:
        obj["diff"]["width"] = "input_smaller"
    return obj


def find_transforms(input_shapes, output_shapes):
    processed_info = []
    for i in range(len(input_shapes)):
        inp = input_shapes[i]
        out = output_shapes[i]
        in_cont = find_objects(inp)
        in_gen_colors = general_color_stats(in_cont, inp)
        out_cont = find_objects(out)
        out_gen_colors = general_color_stats(out_cont, out)
        dim_diffs = check_dimensions(inp, out)
        comparison_results = compare_shapes(in_cont, out_cont)
        info = {"input": {"shapes": in_cont, "color_stats": in_gen_colors},
                "output": {"shapes": out_cont, "color_stats": out_gen_colors},
                "diffs": {"dimensions": dim_diffs, "shapes": comparison_results}}
        processed_info.append(info)

    variability_stats = check_variability(input_shapes, output_shapes)

    # Conditions to check:
    # 1. Relation of input to output dimensions - smaller, same, bigger
    # 2. For each input shape, how does it appear in output - absent, partial, distorted, present
    # (single/many, rotated/moved/scaled/mirrored/inverted, colored, generalized), overlapping, overflowing edge
    # 3. For each input color, number of shapes/pixels, stats
    # 4. Variability of inputs and outputs - pay attention to shapes that are similar by color, position, shape
    # differences may be in those or rotation/scale/generality/symmetry

    # Take the first example and form hypotheses about transformations:
    info = processed_info[0]
    possible_transforms = []

    # if in/out dimensions are different and 2 input shapes are present in output, check their relative positions

    # Transformations options to select from:

    # 1. Overlap:
    # Parameters: priorities list, positioning
    # Changes: one shape is partially visible, the other is moved partially over
    #
    # Recognize: one shape is moved within the other's bounding rect, the other is partially visible
    # Compare: Check what shape moves and in what position relative to another shape (while comparing
    # output shapes to input shapes, consider overlapping possibility - compare_with_overlap)
    # Perform: Move one shape onto another in proper position

    # 2. Decomposition:
    # Parameters:
    # array of actions by h or v or d line (including adding delimiters between them (each third?))
    # coordinates (absolute or relative) play the role in addressing lines or their groups or sections
    # Changes: relative positions of pixels in shape change, their number doesn't
    # May lines be removed?
    #
    # Recognize: Check the number and color stats, bounding rect of the result is bigger Compare: By-line (hvd)
    # comparison (ignoring possible gaps or delimiters), fix top line and determine transformation parameters by each
    # line (if repetitions are spotted, express params as cycles)

    # 3. Crop - output is smaller than input, output is fully contained in input (or may be result of other
    # transforms of input), in-place crop - everything outside turns background (any border shape allowed)

    # 4. adding dimensions: delimiters, edge, cells
    #  Parameters:
    #  what is added - dimensions (h and/or v)
    #  where added - position of input

    # 5. scale: output is smaller or bigger than input, each cell of smaller (sub)array corresponds to NxK cells of
    # bigger (sub)array, stretch is scale in one dimension
    # Parameters: what is scaled - the whole input (its version
    # after transform), shape
    # what ratio - it may be suggested by numerical properties - dimensions,
    # number of colors/shapes, number of pixels, etc.
    # uniform - same ratio for x and y or two different ones
    # ratio 1 - for comparisons

    # 6. move: input is the same as output, at least one shape changes position and is distinct from background (
    # puzzle/ move pieces to anchor point/corner pattern/compactify)
    #
    #  Parameters:
    #  what is moved - shape by color, rarity, coordinate, "all but ...", puzzle pieces

    # where is it moved - fixed, dependent, to shape/edge, to align, to same color line, to fetus cells, to join by
    # anchor cell, to rectangle in center, to fill holes, to marked area, as bridge suggests, to sort pieces, delimiter
    #
    # circular array idea - ?
    # for what distance is it moved - fixed, dependent, to shape/edge, to align,
    # to fetus cells, to join by anchor cell, by given sequence step

    # 7. coloring: input is the same as output, at least one shape changes color (without blending with background) (
    # inverting 2-colors, reversing sequence, line through, line zigzag, breakdown, mask, shape expand,
    # spiral/snake/con-centric pattern)
    #
    #  Parameters:
    #
    # what is colored (may be several shapes) - hv or hvd connected groups, background or blank area according to
    # pattern (solid color, tile, mirror, sequence, spiral, snake, concentric, etc.), connect same color pixels (or
    # corners of big cells), sorted groups, relative cells, (imaginary) linethrough (cell or rect center) (through
    # different backgrounds) and intersections, directed rays (curving around blockers), breakdown groups,
    # nearest cells, concave shapes, mask at fetus cells, grid (top left) cells, meeting lines of different colors
    # and meeting point, max possible lines in irregular area, projections to shape, hv-zigzags, pieces falling on
    # marker rect, making holes or fills (vanishing edges), cropped shape to marker color, vacant rects among noise,
    # gaps in lines, inside + gap + rays, completing example shape at fetus points, ray + reflection, crop marker to
    # shape color, rect by corner cells, complete rect, min max cells in irregular grid, quarters by mask,
    # (water) fill, reverse color sequence, inscribed rect, squeezed line samples, straightened shapes in order or by
    # stats, moved shape, remnants of old background after crop, one of many colors, shapes with holes,
    # irregular half of shape, each odd column from the end, bridges, rects to color of projecting arkers,
    #
    #  which color - fixed, dependent, mapping

    # 8. removal: input is the same as output, at least one shape changes color blending into background
    #  Parameters:
    #  what is removed - shapes, noise - quantifiers "everything except", "all with this property", etc.

    # 9. rotate: NxK may turn into KxN (90 or 270) or NxK (180), respective rotation of (sub)array is observed
    #  Parameters:
    #  what is rotated
    #  what angle

    # 10. mirror: (in-place) shape is reflected using some axis into free space without changing dimensions,
    # (generative) shape is reflected using some axis adding to dimensions, initial shape may be mirrored in
    # different directions, the result may be mirrored again, symmetry is observed
    #
    #  Parameters:
    #  what is mirrored - the whole or shape
    #  what axis - row/column or between, diagonal
    #  keep original?

    # 11. replicate: similar to "move" but initial shape stays and many copies are possible, fetus/anchor points,
    # grid with delimiters, sequence
    #
    #  Parameters:
    #  what is copied - shape(s)
    #
    # where/how to copy - sequence (start, direction(s), step, stop condition, edge partiality), fetus cells, grid,
    # fixed, dependent anchor point

    # 12. replace: at least one shape vanishes and a different one appears, what with what? signal to symbol:
    # variability of input/output
    #
    #  Parameters:
    #  what is replaced - shape1
    #  with what - shape2
    #  mapping

    # 13. logical operations: input is bigger, what sections to overlay (recognize and ignore delimiter),
    # color or background?
    #
    #  Parameters:
    #  what shapes are overlapped (delimiter)
    #  background color or not
    #  what operation
    #  two cells colors and output color are all unique but the same
    #  single color input - operation applies to corners of output size

    # 14. stack: input is bigger, occlusion, what color is on top in all examples, what color is only occluded by the
    # top color
    #
    #  Parameters:
    #  what shapes are stacked (delimiter)
    #  what order

    # 15. min model: replace many with one (extreme generalization), the same set of colors
    #  Parameters:
    #  output format - may be extended or one line

    # 16. pattern restore: copy tile or mirror or radial section where pattern is not missing - recognize tiled or
    # symmetric picture in output or recognize monocolor rectangle of the output size in input
    #
    #  Parameters:
    #  tile or mirror
    #  where to find missing pixels
    #  missing piece

    pass


def check_variability(inputs, outputs):
    results = {}
    #
    return results


# 2D algo:
# find "good" (representative, without blanks to be filled) horizontal line, run 1D on that line
# check "cycles" vertically to find vertical "step"

# 1. Get dimensions of input and output of the first example

train_inputs, train_outputs, test_inputs, test_outputs = process_challenge("0a2355a6", evaluation_challenges,
                                                                           evaluation_solutions)
# print([p.shape for p in train_inputs])
# print([p.shape for p in train_outputs])
# for idx, x in enumerate(train_inputs):
# print(x)
# print(train_outputs[idx])

# 2. Analyze shapes in both - consider hvd-shapes against background -
# their set will have subsets (same color hv-shapes, hv-shapes, same color hvd-shapes,
# lines). If input or output is up to 4x4 - consider the whole array as "cell"
# (collection of non-connected shapes - constellation). Try to characterize input and
# output states - via objects and their relative positions. (A shape cannot be
# "biggest" or "rarest" if we do not know about other shapes, a pixel in a shape
# may have additional properties compared to a single pixel). Separate collections
# of similar shapes or same color shapes. Calculate various properties and statistics
# (properties of collections). Higher order structures provide more options for
# transformations.
shapes = find_shapes(train_inputs[3], True, True)
# print(shapes)
shapes2 = find_objects(train_inputs[3])
print(shapes2)

shapes3 = find_objects(train_outputs[3])
print(shapes3)

find_transforms(train_inputs, train_outputs)


def cog_model(train_inputs, train_outputs, test_inputs,
              test_outputs=None):  # again because test challenges don't have solutions
    computed_answers = []

    # 1. Get dimensions of input and output of the first example.

    # 2. Analyze shapes in both - consider hvd-shapes against background - their set will have subsets (same color
    # hv-shapes, hv-shapes, same color hvd-shapes, lines). If input or output is up to 4x4 - consider the whole array
    # as "cell" (collection of non-connected shapes - constellation). Try to characterize input and output states -
    # via objects and their relative positions. (A shape cannot be "biggest" or "rarest" if we do not know about
    # other shapes, a pixel in a shape may have additional properties compared to a single pixel). Separate
    # collections of similar shapes or same color shapes. Calculate various properties and statistics (properties of
    # collections). Higher order structures provide more options for transformations.

    # 3. Check how objects/colors from smaller (input if equal) one figure in the other one.

    # 3a. Check for objects/colors across inputs in all examples in terms of variability (shapes, colors, positions,
    # rotations). Check the same for outputs.

    # 4. Based on observed differences, form a set of possible transformations and determine their sequence. (It may be
    # a set of transformations to finalize later).

    # 5. For each transformation, for each of their parameters form a set of hypotheses on how that parameter is
    # determined. Sorting shapes by parameter may help to figure out matches or uniqueness.

    # 6. Go through all the other examples and determine which hypotheses are correct. If all fail, go with the ones
    # correct (or closest to "truth") in most cases.

    # 7. Prepare the framework transformation - indicating primitive transformations in proper sequence and
    # parameters for each. Form the Expecto class for test input - what shapes to expect, what shapes to pay
    # attention to, what parameters to determine, how to calculate output dimensions.

    # 8. Apply the framework transformation to test input. Use the Expecto class.

    # for _ in range(len(test_inputs)):
    # answers.append(answer)

    # To be deleted later ************

    avg_shape = np.ceil(np.array([np.array(p.shape) for p in train_outputs]).mean(0)).astype(int)
    for _ in range(len(test_inputs)):
        computed_answers.append(np.random.randint(0, 10, size=avg_shape))

    # End of section to delete *******

    return computed_answers


answers = cog_model(train_inputs, train_outputs, test_inputs, test_outputs=None)


# print(answers)


def get_score(model_answers, real_answers):
    total_score = 0
    for i, answer in enumerate(model_answers):
        # check if both shapes are same, because sometimes we get different shapes
        if answer.shape == real_answers[i].shape:
            score = (answer[i] == real_answers[i]).all()
            if score:
                total_score += 1
            else:
                continue
        else:
            continue

    return int(total_score / len(real_answers))


get_score(answers, test_outputs)

get_score(test_outputs, test_outputs)

ids_evaluation = list(evaluation_challenges)

total_score2 = 0

for i2, challenge_id in enumerate(tqdm(ids_evaluation)):
    train_inputs, train_outputs, test_inputs, test_outputs = process_challenge(challenge_id, evaluation_challenges,
                                                                               evaluation_solutions)

    answers = cog_model(train_inputs, train_outputs, test_inputs, test_outputs)

    total_score2 += get_score(answers, test_outputs)

    # print(f"\ntotal_score: {total_score / (i2 + 1):5f}\n")

ids_test = list(test_challenges)

attempts = []
for i2, challenge_id in enumerate(tqdm(ids_test)):
    train_inputs, train_outputs, test_inputs = process_challenge(challenge_id, test_challenges)

    # attempt1:
    answers1 = cog_model(train_inputs, train_outputs, test_inputs)

    # attempt2:
    answers2 = cog_model(train_inputs, train_outputs, test_inputs)

    attempts.append([answers1, answers2])

for i2, id_test in enumerate(tqdm(ids_test)):
    for dict_attempt in sample_submission[id_test]:
        for ati in range(len(attempts[i2][0])):
            dict_attempt['attempt_1'] = attempts[i2][0][ati].tolist()
            dict_attempt['attempt_2'] = attempts[i2][1][ati].tolist()

# type(sample_submission)

with open("submission.json", "w") as outfile:
    json.dump(sample_submission, outfile)
