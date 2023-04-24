from math import sqrt, pow
from collections import Counter


def euclideanDistance(point1, point2):
    sumSquaredDistance = 0

    # euclidean distance
    for i in range(len(point1)):
        sumSquaredDistance += pow(point2[i] - point2[i], 2)

    return sqrt(sumSquaredDistance)


def mode(labels):
    # return highest freq
    return Counter(labels).most_common(1)[0][0]


def knn(dataset, query, k):
    # first define our distances list
    neighborDistanceIndex = []

    for index, dataPoint in enumerate(dataset):
        distance = euclideanDistance(dataPoint[:-1], query)
        neighborDistanceIndex.append((distance, index))

    # sort the list
    sortedNeighborDistanceIndex = sorted(neighborDistanceIndex)

    # get first k neighbors
    kNearestDistanceIndex = sortedNeighborDistanceIndex[:k]

    kNearestLabels = [dataset[i][-1] for distance, i in kNearestDistanceIndex]

    return kNearestDistanceIndex, mode(kNearestLabels)


def main():
    # initialize a random query
    # col 0 = humidity
    # col 1 = pressure
    # col 2 = output (1->rain, 0->no rain)
    query = [13, 89]

    dataset = [
        [22, 67, 1],
        [23, 58, 1],
        [22, 71, 1],
        [18, 93, 1],
        [19, 86, 1],
        [25, 47, 0],
        [27, 41, 0],
        [29, 39, 0],
        [31, 34, 0],
        [45, 27, 0]
    ]

    kNearestNeighbors, prediction = knn(dataset, query, 3)
    print("Output of KNN:")
    if prediction == 1:
        print("Rain")
    else:
        print("No rain")


if __name__ == '__main__':
    main()
