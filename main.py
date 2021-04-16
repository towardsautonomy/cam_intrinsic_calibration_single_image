# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_LINE_EQUATIONS
    points - a list of two points on a line as [[x1, y1], [x2, y2]]
Returns:
    coefficients a, b, c of line ax + by + c = 0
'''
def compute_line_equation(points):
    pt1 = points[0]
    pt2 = points[1]
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    # compute intercept by substituting one of the points in 
    # equation y = slope*x + intercept
    intercept = pt2[1] - slope*pt2[0]

    # get line equation coefficients
    a = -slope
    b = 1.0
    c = -intercept
    return a, b, c

'''
COMPUTE_POINT_OF_INTERSECTION
    line1 - defined by its coefficients [a1, b1, c1]
    line2 - defined by its coefficients [a2, b2, c2]
Returns:
    point of intersection (x, y)
'''
def compute_point_of_intersection(line1, line2):
    # ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    (a1, b1, c1) = line1
    (a2, b2, c2) = line2
    x = (-c2 + c1) / (-a1 + a2)
    y = (-a1 * x) - c1
    return (x, y)

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # compute line equations for two lines
    a1, b1, c1 = compute_line_equation([points[0], points[1]])
    a2, b2, c2 = compute_line_equation([points[2], points[3]])

    # compute point of intersection
    vanishing_point = compute_point_of_intersection((a1, b1, c1), (a2, b2, c2))
    assert np.dot([vanishing_point[0],vanishing_point[1],1.0], np.transpose([a1, b1, c1])) == 0.0, \
                    'Dot product of point and line must be zero'
    return vanishing_point

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # form equations A.w = 0 with 4 constraints of omega (w)
    # A = np.zeros((vanishing_points.shape[0], 4), dtype=np.float32) 
    A = []
    for i, point_i in enumerate(vanishing_points):
        for j, point_j in enumerate(vanishing_points):
            if i != j and j > i:
                point_i_homogeneous = [point_i[0],point_i[1],1.0]
                point_j_homogeneous = [point_j[0],point_j[1],1.0]
                A.append([point_i_homogeneous[0]*point_j_homogeneous[0]+point_i_homogeneous[1]*point_j_homogeneous[1], \
                          point_i_homogeneous[0]*point_j_homogeneous[2]+point_i_homogeneous[2]*point_j_homogeneous[0], \
                          point_i_homogeneous[1]*point_j_homogeneous[2]+point_i_homogeneous[2]*point_j_homogeneous[1], \
                          point_i_homogeneous[2]*point_j_homogeneous[2]])
    A = np.array(A, dtype=np.float32)
    u, s, v_t = np.linalg.svd(A, full_matrices=True)
    # 4 constraints of omega (w) can be obtained as the last column of v or last row of v_transpose
    w1, w4, w5, w6 = v_t.T[:,-1]
    # form omega matrix
    w = np.array([[w1, 0., w4],
                  [0., w1, w5],
                  [w4, w5, w6]])
    # w = (K.K_transpose)^(-1)
    # K can be obtained by Cholesky factorization followed by its inverse
    K_transpose_inv = np.linalg.cholesky(w)
    K = np.linalg.inv(K_transpose_inv.T)
    # divide by the scaling factor
    K = K / K[-1, -1]

    # return intrinsic matrix
    return K

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # compute vanishing line from first pair of vanishing points
    vanishing_line1 = np.array(compute_line_equation(vanishing_pair1)).transpose()
    # compute vanishing line from second pair of vanishing points
    vanishing_line2 = np.array(compute_line_equation(vanishing_pair2)).transpose()
    # compute omega inverse
    w_inv = np.dot(K, K.transpose())
    # compute angle between these two planes
    l1T_winv_l2 = np.dot(vanishing_line1.transpose(), np.dot(w_inv, vanishing_line2))
    sqrt_l1T_winv_l1 = np.sqrt(np.dot(vanishing_line1.transpose(), np.dot(w_inv, vanishing_line1)))
    sqrt_l2T_winv_l2 = np.sqrt(np.dot(vanishing_line2.transpose(), np.dot(w_inv, vanishing_line2)))
    theta = np.arccos(l1T_winv_l2 / np.dot(sqrt_l1T_winv_l1, sqrt_l2T_winv_l2))
    # convert the angle between planes to degrees and return
    return np.degrees(theta)

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    ## estimate real-world direction vectors given vanishing points
    # first image
    d1i = []
    for v1i in vanishing_points1:
        # vanishing point (v) and 3-dimensional direction vector (d) are related as [d = K.v]
        v1i_homogeneous = np.array([v1i[0], v1i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v1i_homogeneous.T)
        d1i.append(KinvV / np.sqrt(KinvV[0]**2 + KinvV[1]**2 + KinvV[2]**2)) # normalize to make sure you obtain a unit vector
    d1i = np.array(d1i)
    # second image
    d2i = []
    for v2i in vanishing_points2:
        # vanishing point (v) and 3-dimensional direction vector (d) are related as [d = K.v]
        v2i_homogeneous = np.array([v2i[0], v2i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v2i_homogeneous.T)
        d2i.append(KinvV / np.sqrt(KinvV[0]**2 + KinvV[1]**2 + KinvV[2]**2)) # normalize to make sure you obtain a unit vector
    d2i = np.array(d2i)
    
    # the directional vectors in image 1 and image 2 are related by a rotation, R i.e. [d2i = R.d1i] => [R = d2i.d1i_inverse]
    R = np.dot(d2i.T, np.linalg.inv(d1i.T))
    return R

if __name__ == '__main__':
    # Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))