# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Samples views from a sphere.

import math
import numpy as np
import transform
import inout

def fibonacci_sampling(n_pts, radius=1):
    '''
    Returns an arbitrary odd number of almost equidistant points from the
    Fibonacci lattice on a unit sphere.

    Latitude (elevation) represents rotation angle around X axis.
    Longitude (azimuth) represents rotation angle around Z axis

    Ref:
    [1] https://arxiv.org/pdf/0912.4540.pdf
    [2] http://stackoverflow.com/questions/34302938/map-point-to-closest-point-on-fibonacci-lattice
    [3] http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    [4] https://www.openprocessing.org/sketch/41142

    :param n_pts: Number of required points (an odd number).
    :param radius: Radius of the view sphere.
    :return: List of 3D points on the sphere surface.
    '''

    assert(n_pts % 2 == 1) # Needs to be an odd number [1]
    n_pts_half = int(n_pts / 2)

    phi = (math.sqrt(5.0) + 1.0) / 2.0 # Golden ratio
    phi_inv = phi - 1.0
    ga = 2.0 * math.pi * phi_inv # Complementary of golden angle

    pts = []
    for i in range(-n_pts_half, n_pts_half + 1):
        lat = math.asin((2 * i) / float(2 * n_pts_half + 1))
        lon = (ga * i) % (2 * math.pi)
        # print(str(i + n_pts_half + 1) + ' (lat, lon): ' + str(lat) + ' ' + str(lon))

        # Convert the latitude and longitude angles to 3D coordinates
        s = math.cos(lat) * radius
        x, y, z = math.cos(lon) * s, math.sin(lon) * s, math.tan(lat) * s
        pts.append([x, y, z])

        # Calculate rotation matrix and translation vector
        # Note: lat,lon=0,0 is a camera looking to the sphere center from
        # (-radius, 0, 0) in the world (i.e. sphere) coordinate system
        # pi_half = 0.5 * math.pi
        # alpha_x = -lat - pi_half
        # alpha_z = lon + pi_half
        # R_x = transform.rotation_matrix(alpha_x, [1, 0, 0])[:3, :3]
        # R_z = transform.rotation_matrix(alpha_z, [0, 0, 1])[:3, :3]
        # R = np.linalg.inv(R_z.dot(R_x))
        # t = -R.dot(np.array([x, y, z]).reshape((3, 1)))

    return pts

def hinter_sampling(min_n_pts, radius=1):
    '''
    Sphere sampling based on refining icosahedron as described in:
    Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
    Local Patches with a Simple Linear Classifier, BMVC 2008

    :param min_n_pts: Minimum required number of points on the whole view sphere.
    :param radius: Radius of the view sphere.
    :return: 3D points on the sphere surface and a list that indicates on which
             refinement level the points were created.
    '''

    # Get vertices and faces of icosahedron
    a, b, c = 0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0
    pts = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
           (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b), (-c, a, b)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
             (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]

    # Refinement level on which the points were created
    pts_level = [0 for _ in range(len(pts))]

    ref_level = 0
    while len(pts) < min_n_pts:
        ref_level += 1
        edge_pt_map = {} # Mapping from an edge to a newly added point on that edge
        faces_new = [] # New set of faces

        # Each face is replaced by 4 new smaller faces
        for face in faces:
            pt_inds = list(face) # List of point IDs involved in the new faces
            for i in range(3):
                # Add a new point if this edge hasn't been processed yet,
                # or get ID of the already added point.
                edge = (face[i], face[(i + 1) % 3])
                edge = (min(edge), max(edge))
                if edge not in edge_pt_map.keys():
                    pt_new_id = len(pts)
                    edge_pt_map[edge] = pt_new_id
                    pt_inds.append(pt_new_id)

                    pt_new = 0.5 * (np.array(pts[edge[0]]) + np.array(pts[edge[1]]))
                    pts.append(pt_new.tolist())
                    pts_level.append(ref_level)
                else:
                    pt_inds.append(edge_pt_map[edge])

            # Replace the current face with 4 new faces
            faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
                          (pt_inds[3], pt_inds[1], pt_inds[4]),
                          (pt_inds[3], pt_inds[4], pt_inds[5]),
                          (pt_inds[5], pt_inds[4], pt_inds[2])]
        faces = faces_new

    # Project the points to a sphere
    pts = np.array(pts)
    pts *= np.reshape(radius / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))

    # Collect point connections
    pt_conns = {}
    for face in faces:
        for i in range(len(face)):
            pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
            pt_conns[face[i]].add(face[(i + 2) % len(face)])

    # Order the points - starting from the top one and adding the connected points
    # sorted by azimuth
    top_pt_id = np.argmax(pts[:, 2])
    pts_ordered = []
    pts_todo = [top_pt_id]
    pts_done = [False for _ in range(pts.shape[0])]

    def calc_azimuth(x, y):
        two_pi = 2.0 * math.pi
        return (math.atan2(y, x) + two_pi) % two_pi

    while len(pts_ordered) != pts.shape[0]:
        # Sort by azimuth
        pts_todo = sorted(pts_todo, key=lambda i: calc_azimuth(pts[i][0], pts[i][1]))
        pts_todo_new = []
        for pt_id in pts_todo:
            pts_ordered.append(pt_id)
            pts_done[pt_id] = True
            pts_todo_new += [i for i in pt_conns[pt_id]] # Find the connected points

        # Points to be processed in the next iteration
        pts_todo = [i for i in set(pts_todo_new) if not pts_done[i]]

    # Re-order the points and faces
    pts = pts[np.array(pts_ordered), :]
    pts_level = [pts_level[i] for i in pts_ordered]
    pts_order = np.zeros((pts.shape[0],))
    pts_order[np.array(pts_ordered)] = np.arange(pts.shape[0])
    for face_id in range(len(faces)):
        faces[face_id] = [pts_order[i] for i in faces[face_id]]

    # import inout
    # inout.save_ply('output/hinter_sampling.ply', pts=pts, faces=np.array(faces))

    return pts, pts_level

def sample_views(min_n_views, radius=1,
                 azimuth_range=(0, 2 * math.pi),
                 elev_range=(-0.5 * math.pi, 0.5 * math.pi)):
    '''
    Viewpoint sampling from a view sphere.

    :param min_n_views: Minimum required number of views on the whole view sphere.
    :param radius: Radius of the view sphere.
    :param azimuth_range: Azimuth range from which the viewpoints are sampled.
    :param elev_range: Elevation range from which the viewpoints are sampled.
    :return: List of views, each represented by a 3x3 rotation matrix and
             a 3x1 translation vector.
    '''

    # Get points on a sphere
    if True:
        pts, pts_level = hinter_sampling(min_n_views, radius=radius)
    else:
        pts = fibonacci_sampling(min_n_views + 1, radius=radius)
        pts_level = [0 for _ in range(len(pts))]

    views = []
    for pt in pts:
        # Azimuth from (0, 2 * pi)
        azimuth = math.atan2(pt[1], pt[0])
        if azimuth < 0:
            azimuth += 2.0 * math.pi

        # Elevation from (-0.5 * pi, 0.5 * pi)
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        # if hemisphere and (pt[2] < 0 or pt[0] < 0 or pt[1] < 0):
        if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
                elev_range[0] <= elev <= elev_range[1]):
            continue

        # Rotation matrix
        # The code was adopted from gluLookAt function (uses OpenGL coordinate system):
        # [1] http://stackoverflow.com/questions/5717654/glulookat-explanation
        # [2] https://www.opengl.org/wiki/GluLookAt_code
        f = -np.array(pt) # Forward direction
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0]) # Up direction
        s = np.cross(f, u) # Side direction
        if np.count_nonzero(s) == 0:
            # f and u are parallel, i.e. we are looking along or against Z axis
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f) # Recompute up
        R = np.array([[s[0], s[1], s[2]],
                      [u[0], u[1], u[2]],
                      [-f[0], -f[1], -f[2]]])

        # Convert from OpenGL to OpenCV coordinate system
        R_yz_flip = transform.rotation_matrix(math.pi, [1, 0, 0])[:3, :3]
        R = R_yz_flip.dot(R)

        # Translation vector
        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({'R': R, 't': t})

    return views, pts_level

def save_vis(path, views, views_level=None):
    '''
    Creates a PLY file visualizing the views.

    :param path: Path to output PLY file.
    :param views: Views as returned by sample_views().
    :param views_level: View levels as returned by sample_views().
    :return: -
    '''
    # Visualization (saved as a PLY file)
    pts = []
    normals = []
    colors = []
    for view_id, view in enumerate(views):
        R_inv = np.linalg.inv(view['R'])
        pts += [R_inv.dot(-view['t']).squeeze(),
                # R_inv.dot(np.array([[0.01, 0, 0]]).T - view['t']).squeeze(),
                # R_inv.dot(np.array([[0, 0.01, 0]]).T - view['t']).squeeze(),
                # R_inv.dot(np.array([[0, 0, 0.01]]).T - view['t']).squeeze()
                ]

        normal = R_inv.dot(np.array([0, 0, 1]).reshape((3, 1)))
        normals += [normal.squeeze(),
                    # np.array([0, 0, 0]),
                    # np.array([0, 0, 0]),
                    # np.array([0, 0, 0])
                    ]

        if views_level:
            intens = (255 * views_level[view_id]) / float(max(views_level))
        else:
            intens = 255 * view_id / float(len(views))
        colors += [[intens, intens, intens],
                   # [255, 0, 0],
                   # [0, 255, 0],
                   # [0, 0, 255]
                   ]

    inout.save_ply(path,
                   pts=np.array(pts),
                   pts_normals=np.array(normals),
                   pts_colors=np.array(colors))

if __name__ == '__main__':

    # Example of sampling views from a view sphere.
    min_n_views = 1000
    radius = 1
    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)

    # Sample views
    views, views_level = sample_views(min_n_views, radius, azimuth_range, elev_range)
    print('Sampled views: ' + str(len(views)))

    #out_views_vis_path = '../output/view_sphere.ply'
    #save_vis(out_views_vis_path, views)


# Unfinished implementation of sphere sampling used by TUD:
#
# def sample_views_dresden(
#     elev_steps=1,
#     azimuth_steps=1,
#     cell_samples=1,
#     inplane_steps=1,
#     inplane_max_angle=1,
#     z_shift=1
# ):#
#     def slice_sample_angle(x_min, x_max, x_last):
#         if x_last > x_max or x_last < x_min:
#             x_last = random.uniform(x_min, x_max)
#         y = math.cos(x_last)
#         y_sample = random.uniform(0, y)
#         x_limit = min(math.acos(y_sample), x_max)
#         x_sample = random.uniform(x_min, x_limit)
#         return x_sample
#
#     elev_angle = -1.0
#     for elev in range(elev_steps):
#         for azimuth in range(azimuth_steps):
#             for inplane in range(inplane_steps):
#                 for cell_id in range(cell_samples):
#                     start = math.asin(elev / float(elev_steps))
#                     end = math.asin((elev + 1) / float(elev_steps))
#
#                     # Sample elevation
#                     elev_angle = slice_sample_angle(start, end, elev_angle)
#
#                     # Sample azimuth
#                     extent = (2 * math.pi) / float(azimuth_steps)
#                     azimuth_angle = azimuth * extent + random.uniform(0, extent)
#
#                     # Sample in-plane rotation
#                     extent = inplane_max_angle * 2.0 / float(inplane_steps)
#                     inplane_angle_start = -inplane_max_angle + extent * inplane
#                     inplane_angle_end = inplane_angle_start + extent
#                     inplane_angle = random.uniform(inplane_angle_start, inplane_angle_end)
#
#                     rvI = [0, 0, inplane_angle, 0, 0, 0]
#                     #6DPose hIn(rvI);
#
#                     rv_ = [elev_angle, 0, 0, 0, 0, 0]
#                     #6DPose hAng(rv_);
#
#                     rv = [0, azimuth_angle, 0, 0, 0, 0]
#                     #6DPose hAngB(rv);
#
#                     # Combine rotations around axis into one rotation and set translation
#                     #6DPose pose(hIn.getTransformation() * hAng.getTransformation() * hAngB.getTransformation())
#                     #pose.setTranslation(cv::Point3d(0.0, 0.0, z_shift * 1000.0))
