import cv2
import math
import sys

from time import sleep


def add_if_white(cnt, cl):
    if cl == 255:
        cnt += 1
    return cnt


def skeletonize(thresh):
    # do distance transform
    dist = cv2.distanceTransform(thresh, distanceType=cv2.DIST_L2, maskSize=5)

    # set up cross for tophat skeletonization
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = cv2.morphologyEx(dist, cv2.MORPH_TOPHAT, kernel)

    # threshold skeleton
    ret, skel = cv2.threshold(skel, 0, 255, 0)

    return skel


class Points:
    def __init__(self, points):
        self.points = points
        self.n = len(points)

        self.groups = [-1 for i in range(self.n)]
        self.eps_distance = 10

    # Recursive function, similar to DFS
    def merge_points(self, index, group_id):
        self.groups[index] = group_id
        for idx, coord in enumerate(self.points):
            if self.groups[idx] != -1: continue

            if abs(coord[0] - self.points[index][0]) <= self.eps_distance and \
                    abs(coord[1] - self.points[index][1]) <= self.eps_distance:
                self.merge_points(idx, group_id)

    # Simplified call function
    def calc(self):
        cl = 0
        for i in range(self.n):
            if self.groups[i] == -1:
                self.merge_points(i, cl)
                cl += 1

        ans = []
        for id in range(cl):
            avgx, avgy, cnt = 0, 0, 0
            for i in range(self.n):
                if self.groups[i] == id:
                    avgx += self.points[i][0]
                    avgy += self.points[i][1]
                    cnt += 1
            ans.append([avgx // cnt, avgy // cnt])

        return ans


# This class is practically the same as Points, however it works with angles, so I've created a separate class
# TODO: consider merging these two classes together
class Angles:
    def __init__(self, angles):
        self.angles = angles
        self.n = len(angles)

        self.groups = [-1 for i in range(self.n)]
        self.eps_angle = 10

    # Recursive function, similar to DFS
    def merge_angles(self, index, group_id):
        self.groups[index] = group_id
        for idx, ang in enumerate(self.angles):  # Iterate over all angles
            if self.groups[idx] != -1 or idx == index: continue  # Already assigned group id

            if abs(ang - self.angles[index]) <= self.eps_angle or \
                    abs(ang + 360 - self.angles[index]) <= self.eps_angle or \
                    abs(ang - 360 - self.angles[index]) <= self.eps_angle:
                self.merge_angles(idx, group_id)

    # Simplified call function
    def calc(self):
        cl = 0  # Number of groups
        for i in range(self.n):
            if self.groups[i] == -1:
                self.merge_angles(i, cl)  # Call recursive function from every angle that is not assigned to a group yet
                cl += 1

        ans = []
        for id in range(cl):
            x, y = 0, 0  # x, y to calc the average
            for i in range(self.n):
                if self.groups[i] == id:
                    rad_angle = self.angles[i] / 180 * math.pi
                    x += math.cos(rad_angle)
                    y += math.sin(rad_angle)

            avg_angle = int(math.atan2(y, x) / math.pi * 180)
            if avg_angle < 0: avg_angle += 360
            ans.append(avg_angle)

        return ans


def count_branches(resized_skel, x, y, radius, eps):
    angles_dict = {}
    height, width = resized_skel.shape

    for r in range(radius - 1, radius + 1):
        for angle in range(0, 360, 2):
            rad_angle = angle * math.pi / 180  # Convert to radians

            # Coordinates of the point on the circle
            _x = int(x + r * math.cos(rad_angle))
            _y = int(y + r * math.sin(rad_angle))

            if 0 <= _x < width and 0 <= _y < height:  # in range
                if resized_skel[_y][_x] == 255:
                    angles_dict[_y * height + _x] = angle

    angles = list(angles_dict.values())

    angle_merger = Angles(angles)
    return angle_merger.calc()


# CHeck if there is a consistent branch on all levels
def recursive_filter_branches(all_branches, layer, index):
    if layer + 1 >= len(all_branches):
        return [index]

    for j in range(len(all_branches[layer + 1])):
        if abs(all_branches[layer][index] - all_branches[layer + 1][j]) < 30:
            returned = recursive_filter_branches(all_branches, layer + 1, j)
            if returned: return returned + [index]

    return []


# Main function
def parse_symbol(path):
    sys.setrecursionlimit(10000)

    img = 255 - cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    org = cv2.imread(path)
    _org = cv2.imread(path)

    # threshold img
    ret, thresh = cv2.threshold(img, img.mean(axis=0).mean(axis=0) + 20, 255, 0)

    # Remove lines
    height, width = thresh.shape
    for y in range(height):
        for x in range(width):
            if y not in range(17, height - 17) or x not in range(17, width - 17):
                thresh[y][x] = 0

    skel = skeletonize(thresh)

    # Resize the drawing to the max size
    rect = cv2.boundingRect(thresh)
    x, y, w, h = rect
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize the working area of the image to the normal size for easier future manipulation
    if h > w:
        resized = cv2.resize(skel[y: y + h, x: x + w], (int(w * (height / h)), height), interpolation=cv2.INTER_AREA)
        _org = cv2.resize(_org[y: y + h, x: x + w], (int(w * (height / h)), height), interpolation=cv2.INTER_AREA)
        org = cv2.resize(org[y: y + h, x: x + w], (int(w * (height / h)), height), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(skel[y: y + h, x: x + w], (width, int(h * (width / w))), interpolation=cv2.INTER_AREA)
        _org = cv2.resize(_org[y: y + h, x: x + w], (width, int(h * (width / w))), interpolation=cv2.INTER_AREA)
        org = cv2.resize(org[y: y + h, x: x + w], (width, int(h * (width / w))), interpolation=cv2.INTER_AREA)

    ret, resized_skel = cv2.threshold(resized, 0, 255, 0)

    # radius = 30  # Radius of the circle from the point
    eps = 50  # Merge angles
    angle_difference = 95  # If 2 branches out of current point, then what should the difference be
    height, width = resized_skel.shape

    # Remove unnecessary points
    # TODO: remove many repeated lines
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if resized_skel[y][x] != 255: continue
            cnt = 0
            cnt = add_if_white(cnt, resized_skel[y + 1][x])
            cnt = add_if_white(cnt, resized_skel[y - 1][x])
            cnt = add_if_white(cnt, resized_skel[y][x + 1])
            cnt = add_if_white(cnt, resized_skel[y][x - 1])

            cnt = add_if_white(cnt, resized_skel[y + 1][x + 1])
            cnt = add_if_white(cnt, resized_skel[y - 1][x - 1])
            cnt = add_if_white(cnt, resized_skel[y + 1][x - 1])
            cnt = add_if_white(cnt, resized_skel[y - 1][x + 1])
            if cnt <= 1:
                resized_skel[y][x] = 0  # Remove if the number of surrounding skeleton pixels is very small

    # Main body
    points = []
    for y in range(height):
        for x in range(width):
            if resized_skel[y][x] == 255:  # if it is a part of line
                cleaned_angles = []
                for rad in range(10, 30, 5):
                    # Check branches with increasing radius
                    cleaned_angles.append(count_branches(resized_skel, x, y, rad, eps))

                # Search through layers and find consistent branches
                avg_angles = []
                for i in range(len(cleaned_angles[0])):  # Iterate over the first layer of angles & call recursively
                    returned = recursive_filter_branches(cleaned_angles, 0, i)
                    if returned:  # Calc average angle of the branch
                        x, y = 0, 0  # x, y to calc the average

                        for j, index_in_layer in enumerate(returned):
                            rad_angle = cleaned_angles[len(cleaned_angles) - j - 1][index_in_layer] / 180 * math.pi  # `returned` is reversed
                            x += math.cos(rad_angle)
                            y += math.sin(rad_angle)

                        avg_angle = int(math.atan2(y, x) / math.pi * 180)
                        if avg_angle < 0: avg_angle += 360

                        avg_angles.append(avg_angle)

                branch_cnt = len(avg_angles)  # Total number of branches coming out of this pixel

                if branch_cnt == 1 or branch_cnt == 1:
                    points.append([x, y])
                    if 30 < y < 100:
                        org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)

                        for layer in range(10, 30, 5):
                            for a in cleaned_angles[layer // 5 - 2]:
                                radian_angle = a * math.pi / 180
                                _x = int(x + layer * math.cos(radian_angle))
                                _y = int(y + layer * math.sin(radian_angle))

                                org = cv2.circle(org, (_x, _y), 0, (0, 0, 255), -1)

                        cv2.imwrite('org.png', org)
                        sleep(3)

                # elif branch_cnt == 2:
                #     diff = abs(avg_angles[0] - avg_angles[1])
                #     if min(diff, 360 - diff) < angle_difference:
                #         points.append([x, y])
                #         org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)

    # Merge multiple points into one
    p = Points(points)
    merged_points = p.calc()

    for elem in merged_points:
        _org = cv2.circle(_org, (int(elem[0]), int(elem[1])), 5, (0, 255, 0), 2)

    # cv2.imwrite("tmp.png", _org)
    # cv2.imshow("resized_skel", _org)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return org, resized_skel, thresh, _org, len(merged_points)
