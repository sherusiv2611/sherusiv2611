import cv2
from scipy.spatial import distance
import numpy as np
import random
from itertools import izip

def compute_ctd(clts):
    return [np.mean(cluster, axis=0) for cluster in clts]

def kmeans(k, ctd, points):
    clts = [[] for _ in range(k)]
    for point in points:
        clts[closest_centroid(point, ctd)].append(point)
    new_ctd = compute_ctd(clts)
    global cl
    cl-=1

    if cl > 1 and not equals(np.array(ctd), np.array(new_ctd)):
        clts = kmeans(k, new_ctd, points)

    return clts

def closest_centroid(point, ctd):
    min_distance = float('inf')
    belongs_to_cluster = None
    for j, centroid in enumerate(ctd):
        dist = distance.sqeuclidean(point, centroid)
        if dist < min_distance:
            min_distance = dist
            belongs_to_cluster = j

    return belongs_to_cluster

def equals(points1, points2):
    if len(points1) != len(points2):
        return False
    for point1, point2 in izip(points1, points2):
        if any(x.all() != y.all() for x, y in izip(points1, points2)):
            return False

    return True


original = cv2.imread("1.jpeg")

cl = 10
cv2.cvtColor(original,cv2.COLOR_BGR2LAB,original)
original_shape = original.shape
data = original.reshape(-1, 3)

k = 5

ctd = random.sample(data, k)
clts = kmeans(k, ctd, data)
new_ctd = compute_ctd(clts)
img = []

for p in data:
    img.append(new_ctd[closest_centroid(p, new_ctd)])
print ("executing")
image  = np.array(img,  dtype=np.uint8).reshape(original_shape)
cv2.cvtColor(image,cv2.COLOR_LAB2BGR, image)
cv2.imshow('3.jpeg', image)
cv2.imwrite('4.jpeg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
