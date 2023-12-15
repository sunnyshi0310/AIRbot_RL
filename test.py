
print("1" in "123")

from robot_tools import CoordinateTools, transformations

print(CoordinateTools.transform_as_matrix([3,0,2],[0,0,0,1]))
print(transformations.euler_matrix(0,0,0))