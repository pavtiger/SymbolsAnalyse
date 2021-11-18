import cv2
import math
import sys
from collections import defaultdict

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
    def __init__(self, angles, radius, eps):
        self.angles = angles
        self.n = len(angles)

        self.groups = [-1 for i in range(self.n)]
        self.eps_angle = eps

    # Recursive function, similar to DFS
    def merge_angles(self, index, group_id):
        self.groups[index] = group_id
        for idx, angle in enumerate(self.angles):  # Iterate over all angles
            if self.groups[idx] != -1 or idx == index: continue  # Already assigned group id

            ang = angle[0]
            if min(abs(ang - self.angles[index][0]), 360 - abs(ang - self.angles[index][0])) <= self.eps_angle:
                self.merge_angles(idx, group_id)

    # Simplified call function
    def calc(self, min_rad, min_amount):
        cl = 0  # Number of groups
        for i in range(self.n):
            if self.groups[i] == -1 and self.angles[i][1] <= min_rad:
                self.merge_angles(i, cl)  # Call recursive function from every angle that is not assigned to a group yet
                cl += 1

        ans = []
        for id in range(cl):
            cnt, x, y, min_rad, avg_rad = 0, 0, 0, 100, 0  # x, y to calc the average
            for i in range(self.n):
                if self.groups[i] == id:
                    rad_angle = self.angles[i][0] / 180 * math.pi
                    x += math.cos(rad_angle)
                    y += math.sin(rad_angle)
                    cnt += 1
                    avg_rad += self.angles[i][1]
                    # min_rad = min(min_rad, self.angles[i][1])

            avg_angle = int(math.atan2(y, x) / math.pi * 180)
            # min_rad = min(min_rad)
            if avg_angle < 0: avg_angle += 360

            # print(min_rad)
            # ans.append(avg_angle)
            if cnt > min_amount: ans.append(avg_angle)
            # if cnt > 3 and min_rad <= 6: ans.append(avg_angle)
            # if cnt > 3 and (avg_rad / cnt) <= 25: ans.append(avg_angle)

        return ans


def count_branches(resized_skel, x, y, radius, eps, org):
    angles_dict = defaultdict(list)
    height, width = resized_skel.shape

    cnt = 0

    for r in range(radius - 1, radius + 1):
        for angle in range(0, 360, 1):
            rad_angle = angle * math.pi / 180  # Convert to radians

            # Coordinates of the point on the circle
            _x = int(x + r * math.cos(rad_angle))
            _y = int(y + r * math.sin(rad_angle))

            # org = cv2.circle(org, (_x, _y), 0, (0, 255, 0), -1)

            if 0 <= _x < width and 0 <= _y < height:  # in range
                if resized_skel[_y][_x] == 255:
                    angles_dict[f'{_y}-{_x}'].append(angle)
                    cnt += 1

    angles = []
    for angles_to_average in angles_dict.values():
        vx, vy = 0, 0
        for angle in angles_to_average:
            rad_angle = angle / 180 * math.pi
            vx += math.cos(rad_angle)
            vy += math.sin(rad_angle)

        avg_angle = int(math.atan2(vy, vx) / math.pi * 180)
        if avg_angle < 0: avg_angle += 360
        angles.append(avg_angle)

    converted = []
    for a in angles:
        converted.append([a, 0])

    angle_merger = Angles(converted, radius, 20)
    ret = angle_merger.calc(100000, 0)
    if len(ret) == 0:
        print(cnt)
        # org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)

    return ret
    # return angles


# Check if there is a consistent branch on all levels
def recursive_filter_branches(all_branches, layer, index, eps, length):
    if layer + 1 >= len(all_branches):
        return [index]

    best = []

    for j in range(len(all_branches[layer + 1])):
        if abs(all_branches[layer][index] - all_branches[layer + 1][j]) < eps:
            returned = recursive_filter_branches(all_branches, layer + 1, j, eps, length + 1)
            if (len(returned) + 1) > len(best):
                best = [index] + returned

    # if layer + 2 < len(all_branches):
    #     for j in range(len(all_branches[layer + 2])):
    #         if abs(all_branches[layer][index] - all_branches[layer + 2][j]) < eps:
    #             returned = recursive_filter_branches(all_branches, layer + 2, j, eps, length + 1)
    #             if (len(returned) + 1) > len(best):
    #                 best = returned + [index]

    return best


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
                cleaned_angles, layered_angles = [], []
                for rad in range(6, 40, 1):  # 4, 40, 2     10, 12, 2
                    # Check branches with increasing radius
                    branches = count_branches(resized_skel, x, y, rad, eps, org)
                    if len(branches) == 0:
                        print(f'{x}, {y} - {rad}')
                        # org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)
                        # count_branches(resized_skel, x, y, rad, eps, org)

                    layered_angles.append(branches)
                    for angle in branches:
                        cleaned_angles.append([angle, rad])

                # Merge similar angles together
                # angle_merger = Angles(cleaned_angles, -1, 50)
                # the_angles = angle_merger.calc(8, 6)

                the_angles = []
                used = {}
                for layer in range(len(layered_angles)):
                    used[layer] = [False for _ in range(len(layered_angles[layer]))]

                for layer in range(0, 3):
                    used[layer] = [False for _ in range(len(layered_angles[layer]))]
                    for index in range(len(layered_angles[layer])):
                        if used[layer][index]: continue

                        ret = recursive_filter_branches(layered_angles, layer, index, 30, 1)
                        if len(ret) > 3:
                            _x, _y = 0, 0
                            for l, ind in enumerate(ret):
                                used[l + layer][ind] = True

                                rad_angle = layered_angles[l + layer][ind] / 180 * math.pi
                                _x += math.cos(rad_angle)
                                _y += math.sin(rad_angle)

                            avg_angle = int(math.atan2(_y, _x) / math.pi * 180)
                            if avg_angle < 0: avg_angle += 360
                            the_angles.append(avg_angle)

                branch_cnt = len(the_angles)  # Total number of branches coming out of this pixel (for the ease of use)

                # cv2.imwrite('org.png', org)
                # sleep(5)

                if branch_cnt >= 3 or branch_cnt == 1:
                    # if 0 < x < 230 and 20 < y < 100:
                    # if 220 < x < 300 and 60 < y < 130:
                    if 100 < x < 250 and 100 < y < 250:
                        points.append([x, y])
                        arr = list(list(zip(*cleaned_angles))[0])
                        arr.sort()
                        org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)
                    else:
                        org = cv2.circle(org, (x, y), 0, (0, 0, 255), -1)

                    # am = Angles(cleaned_angles, -1, 30)
                    # am.calc(8, 6)

                    # for layer in range(5, 40, 1):
                    #     for a in layered_angles[layer // 5 - 2]:
                    #         radian_angle = a * math.pi / 180
                    #         _x = int(x + layer * math.cos(radian_angle))
                    #         _y = int(y + layer * math.sin(radian_angle))
                    #
                    #         org = cv2.circle(org, (_x, _y), 0, (0, 0, 255), -1)

                    # cv2.imwrite('org.png', org)

                # elif branch_cnt == 2:
                    # diff = abs(the_angles[0] - the_angles[1])
                    # if min(diff, 360 - diff) < angle_difference:
                    #     points.append([x, y])
                    #     org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)

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
