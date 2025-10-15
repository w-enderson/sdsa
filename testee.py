from models.sdsa import euclidean_dist, city_block_dist, sqeuclidean_dist
import numpy as np



matrix1 = np.array([[1,1,1,1], [2,2,2,2]])
matrix2 = np.array([[1,2, 1,2], [2,2,1,1]])
print(euclidean_dist(matrix1, matrix2))
print(sqeuclidean_dist(matrix1, matrix2))
print(city_block_dist(matrix1, matrix2))