import numpy as np
import struct


class OpenGLRenderer(object):
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }
    models = {}

    def __init__(self, ply_path):
        self.ply_path = ply_path
        self.model = self.load_ply(self.ply_path)

    def load_ply(self, ply_path):
        """ Loads a 3D mesh model from a PLY file.
        :return: The loaded model given by a dictionary with items:
        'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
        'faces' (mx3 ndarray) - the latter three are optional.
        """
        f = open(ply_path, 'r')

        n_pts = 0
        n_faces = 0
        face_n_corners = 3 # Only triangular faces are supported
        pt_props = []
        face_props = []
        is_binary = False
        header_vertex_section = False
        header_face_section = False

        # Read header
        while True:
            line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
            if line.startswith('element vertex'):
                n_pts = int(line.split()[-1])
                header_vertex_section = True
                header_face_section = False
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
                header_vertex_section = False
                header_face_section = True
            elif line.startswith('element'): # Some other element
                header_vertex_section = False
                header_face_section = False
            elif line.startswith('property') and header_vertex_section:
                # (name of the property, data type)
                pt_props.append((line.split()[-1], line.split()[-2]))
            elif line.startswith('property list') and header_face_section:
                elems = line.split()
                if elems[-1] == 'vertex_indices':
                    # (name of the property, data type)
                    face_props.append(('n_corners', elems[2]))
                    for i in range(face_n_corners):
                        face_props.append(('ind_' + str(i), elems[3]))
                else:
                    print('Warning: Not supported face property: ' + elems[-1])
            elif line.startswith('format'):
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('end_header'):
                break

        # Prepare data structures
        model = {}
        model['pts'] = np.zeros((n_pts, 3), np.float)
        if n_faces > 0:
            model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

        pt_props_names = [p[0] for p in pt_props]
        is_normal = False
        if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
            is_normal = True
            model['normals'] = np.zeros((n_pts, 3), np.float)

        is_color = False
        model['colors'] = np.zeros((n_pts, 3), np.float)
        if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
            is_color = True
            model['colors'] = np.zeros((n_pts, 3), np.float)

        is_texture = False
        if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
            is_texture = True
            model['texture_uv'] = np.zeros((n_pts, 2), np.float)

        formats = { # For binary format
            'float': ('f', 4),
            'double': ('d', 8),
            'int': ('i', 4),
            'uchar': ('B', 1)
        }

        # Load vertices
        for pt_id in range(n_pts):
            prop_vals = {}
            load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                          'red', 'green', 'blue', 'texture_u', 'texture_v']
            if is_binary:
                for prop in pt_props:
                    format = formats[prop[1]]
                    val = struct.unpack(format[0], f.read(format[1]))[0]
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = val
            else:
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(pt_props):
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = elems[prop_id]

            model['pts'][pt_id, 0] = float(prop_vals['x'])
            model['pts'][pt_id, 1] = float(prop_vals['y'])
            model['pts'][pt_id, 2] = float(prop_vals['z'])

            if is_normal:
                model['normals'][pt_id, 0] = float(prop_vals['nx'])
                model['normals'][pt_id, 1] = float(prop_vals['ny'])
                model['normals'][pt_id, 2] = float(prop_vals['nz'])

            if is_color:
                model['colors'][pt_id, 0] = float(prop_vals['red'])
                model['colors'][pt_id, 1] = float(prop_vals['green'])
                model['colors'][pt_id, 2] = float(prop_vals['blue'])

            if is_texture:
                model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
                model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

        # Load faces
        for face_id in range(n_faces):
            prop_vals = {}
            if is_binary:
                for prop in face_props:
                    format = formats[prop[1]]
                    val = struct.unpack(format[0], f.read(format[1]))[0]
                    if prop[0] == 'n_corners':
                        if val != face_n_corners:
                            print('Error: Only triangular faces are supported.')
                            print('Number of face corners: ' + str(val))
                            exit(-1)
                    else:
                        prop_vals[prop[0]] = val
            else:
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(face_props):
                    if prop[0] == 'n_corners':
                        if int(elems[prop_id]) != face_n_corners:
                            print('Error: Only triangular faces are supported.')
                            print('Number of face corners: ' + str(int(elems[prop_id])))
                            exit(-1)
                    else:
                        prop_vals[prop[0]] = elems[prop_id]

            model['faces'][face_id, 0] = int(prop_vals['ind_0'])
            model['faces'][face_id, 1] = int(prop_vals['ind_1'])
            model['faces'][face_id, 2] = int(prop_vals['ind_2'])

        f.close()
        model['pts'] *= 1000.

        return model

    def render(self, pose, K, img_size, render_type='depth'):
        from .opengl_backend import render
        R = pose[:, :3]
        t = pose[:, 3:] * 1000.
        model = self.model
        if render_type == 'depth':
            return render(model, im_size=img_size, K=K, R=R, t=t, clip_near=10, clip_far=10000, mode='depth') / 1000.
        else:
            return render(model, im_size=img_size, K=K, R=R, t=t, clip_near=10, clip_far=10000, mode='rgb')

