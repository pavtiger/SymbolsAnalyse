import cv2
import math


def add_if_white(cnt, cl):
    if cl == 255:
        cnt += 1
    return cnt


def parse_symbol(path):
    img = 255 - cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    org = cv2.imread(path)
    _org = cv2.imread(path)

    # threshold img
    # print(path, img.mean(axis=0).mean(axis=0))
    # ret, thresh = cv2.threshold(img, 90, 255, 0)
    ret, thresh = cv2.threshold(img, img.mean(axis=0).mean(axis=0) + 20, 255, 0)
    # if thresh.mean(axis=0).mean(axis=0) >

    # do distance transform
    dist = cv2.distanceTransform(thresh, distanceType=cv2.DIST_L2, maskSize=5)

    # set up cross for tophat skeletonization
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = cv2.morphologyEx(dist, cv2.MORPH_TOPHAT, kernel)

    # threshold skeleton
    ret, skel = cv2.threshold(skel, 0, 255, 0)

    radius = 6
    eps = 30
    angle_difference = 110
    point_merging_eps = 20
    height, width = skel.shape

    # Remove unnecessary points
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skel[y][x] != 255: continue
            cnt = 0
            cnt = add_if_white(cnt, skel[y + 1][x])
            cnt = add_if_white(cnt, skel[y - 1][x])
            cnt = add_if_white(cnt, skel[y][x + 1])
            cnt = add_if_white(cnt, skel[y][x - 1])

            cnt = add_if_white(cnt, skel[y + 1][x + 1])
            cnt = add_if_white(cnt, skel[y - 1][x - 1])
            cnt = add_if_white(cnt, skel[y + 1][x - 1])
            cnt = add_if_white(cnt, skel[y - 1][x + 1])
            if cnt <= 1:
                skel[y][x] = 0

    intersections = []
    for y in range(10, height - 10):
        for x in range(10, width - 10):
            if skel[y][x] == 255:  # if it is a part of line
                angles_dict = {}

                for r in range(radius - 1, radius + 1):
                    for angle in range(0, 360, 5):
                        rad_angle = angle * math.pi / 180
                        _x = int(x + r * math.cos(rad_angle))
                        _y = int(y + r * math.sin(rad_angle))

                        if 0 <= _x < width and 0 <= _y < height:  # in range
                            if skel[_y][_x] == 255:
                                angles_dict[_y * height + _x] = angle

                cleaned_angles = []
                bad_angles = set()

                angles = list(angles_dict.values())

                for angle in angles:
                    if angle in bad_angles: continue

                    cleaned_angles.append(angle)
                    for close_angle in range(angle - eps, angle + eps):
                        if close_angle < 0:
                            close_angle += 360
                        if close_angle >= 360:
                            close_angle -= 360

                        if close_angle in angles:
                            angle = (angle + close_angle) / 2
                            bad_angles.add(close_angle)

                if len(cleaned_angles) >= 3 or len(cleaned_angles) == 1:
                    intersections.append([x, y])
                    org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)
                    for a in cleaned_angles:
                        rad_angle = a * math.pi / 180
                        _x = int(x + r * math.cos(rad_angle))
                        _y = int(y + r * math.sin(rad_angle))
                        org = cv2.circle(org, (_x, _y), 0, (0, 0, 255), -1)

                elif len(cleaned_angles) == 2:
                    diff = abs(cleaned_angles[0] - cleaned_angles[1])
                    if min(diff, 360 - diff) < angle_difference:
                        intersections.append([x, y])
                        org = cv2.circle(org, (x, y), 0, (0, 255, 0), -1)

    # Merge intersections into one
    for idx, inter in enumerate(intersections):
        a, b = inter
        match = 0
        for other_inter in intersections[idx:]:
            if other_inter == inter:
                continue
            c, d = other_inter
            if abs(c - a) < 15 and abs(d - b) < point_merging_eps:
                match += 1
                intersections[idx] = ((c + a) / 2, (d + b) / 2)
                intersections.remove(other_inter)

        if match == 0:
            intersections.remove(inter)

    for elem in intersections:
        _org = cv2.circle(_org, (int(elem[0]), int(elem[1])), 4, (0, 255, 0), 3)

    return org, skel, thresh, _org, len(intersections)