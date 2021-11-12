import cv2
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image, ImageFilter

from symbol_operate import parse_symbol


# Some geometric functions
def cp(a, b):
    return a[0] * b[1] - b[0] * a[1]


# Calc poly area
def poly_area(poly):
    summ = 0
    for i in range(len(poly)):
        if i == len(poly) - 1:
            # Last point in poly
            summ += cp(poly[i][0], poly[0][0])
        else:
            summ += cp(poly[i][0], poly[i + 1][0])

    return abs(summ) / 2


def find_contours(name, ind):
    image = cv2.imread(f'data/{name}.jpg')  # Load image, grayscale, adaptive threshold
    Path(f'res/{name}').mkdir(parents=True, exist_ok=True)  # Create directory for a person

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to black & white
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 25)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Find contour with max area
    max_cont = 0
    maxx = 0
    for c in cnts:
        area = poly_area(c)
        if area > maxx:
            maxx = area
            max_cont = c

    approx = cv2.approxPolyDP(max_cont, 0.009 * cv2.arcLength(max_cont, True), True)  # ???
    # cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)  # drawing rectangular area of detected box

    # find top left corner of the biggest square
    top_left_corner = 0
    minn = np.sum(approx[0])
    for i, elem in enumerate(approx):
        if np.sum(elem[0]) < minn:
            minn = np.sum(elem[0])
            top_left_corner = i

    # All points are in format [cols, rows]
    pt_A = approx[top_left_corner][0]
    pt_B = approx[(top_left_corner + 1) % 4][0]
    pt_C = approx[(top_left_corner + 2) % 4][0]
    pt_D = approx[(top_left_corner + 3) % 4][0]

    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # cv2.imshow('i', warped)
    # cv2.imwrite(f'normals/{name}.png', warped)

    vertices = []
    # Img size and icon size
    height, width, channels = warped.shape
    h, w = int(height / 8), int(width / 6)
    for i in range(8):  # line
        for j in range(6):  # column
            if i * 6 + j != 23: continue

            a, b, c, d = i * h, (i + 1) * h, j * w, (j + 1) * w
            cv2.imwrite(f'res/{name}/{i * 6 + j}_og.png', warped[a:b, c:d])

            symbol_graph, skeleton, thresh, ans, num_vertices = parse_symbol(f'res/{name}/{i * 6 + j}_og.png')

            cv2.imwrite(f'res/{name}/{i * 6 + j}_og.png', symbol_graph)
            cv2.imwrite(f'res/{name}/{i * 6 + j}.png', ans)
            cv2.imwrite(f'res/{name}/{i * 6 + j}_sk.png', skeleton)
            cv2.imwrite(f'res/{name}/{i * 6 + j}_th.png', thresh)

            # image = Image.open(f'res/{name}/{i * 6 + j}_th.png')
            # image = image.filter(ImageFilter.ModeFilter(size=13))
            # image.save(f'res/{name}/{i * 6 + j}_th.png')

            vertices.append(num_vertices)

    line = {}
    for i in range(96):
        if i >= len(vertices):
            line[str(i)] = 0
        else:
            line[str(i)] = vertices[i]

    data.loc[ind] = line

    data.to_csv(filename, index=False)


# train = [5, 1, 6, 47]
# train_ans = {5: [],
#              1: [4, 5, 3, 3, 4, 7, 0, 4, 6, 0, 5, 2, 3, 4, 7, 4, -1, -1, 4, 6, 4, 2, 3, 4, 6, 7, 1, 6, 2, 8, 3, 4, 5, 6,
#                  3, 8, 9, 7, 5, 4, 3, 3, 5, 4, 7, 6, 8, 6, 9],
#              6: [],
#              47: []}

if __name__ == '__main__':
    filename = 'symbol_data.csv'

    data = pd.DataFrame(columns=[str(i) for i in range(96)])  # Create pandas table
    ind = 0  # Start line

    for q in range(64):  # Iterate over people
        print(q)
        find_contours(str(q + 1), ind)
        ind += 1
        if os.path.isfile(f'data/{str(q + 1)}_.jpg'):  # Check if second page of questionnaire exists
            find_contours(str(q + 1) + '_', ind)
            ind += 1
