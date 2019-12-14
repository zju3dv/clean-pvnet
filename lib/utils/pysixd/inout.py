# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import struct
import itertools
import numpy as np
import scipy.misc
import png
import ruamel.yaml as yaml
# import yaml

# Set representation of the floating point numbers in YAML files
def float_representer(dumper, value):
    text = '{0:.8f}'.format(value)
    return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)
yaml.add_representer(float, float_representer)

def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f, Loader=yaml.CLoader)
        return content

def save_yaml(path, content):
    with open(path, 'w') as f:
        yaml.dump(content, f, Dumper=yaml.CDumper, width=10000)

def load_cam_params(path):
    with open(path, 'r') as f:
        c = yaml.load(f, Loader=yaml.CLoader)
    cam = {
        'im_size': (c['width'], c['height']),
        'K': np.array([[c['fx'], 0.0, c['cx']],
                       [0.0, c['fy'], c['cy']],
                       [0.0, 0.0, 1.0]])
    }
    if 'depth_scale' in c.keys():
        cam['depth_scale'] = float(c['depth_scale'])
    return cam

def load_im(path):
    im = scipy.misc.imread(path)

    # Using PyPNG
    # r = png.Reader(filename=path)
    # im = np.vstack(itertools.imap(np.uint8, r.asDirect()[2]))

    return im

def save_im(path, im):
    scipy.misc.imsave(path, im)

    # Using PyPNG (for RGB)
    # w_rgb = png.Writer(im.shape[1], im.shape[0], greyscale=False, bitdepth=8)
    # with open(path, 'wb') as f:
    #     w_rgb.write(f, np.reshape(im, (-1, 3 * im.shape[1])))

def load_depth(path):
    # PyPNG library is used since it allows to save 16-bit PNG
    r = png.Reader(filename=path)
    im = np.vstack(itertools.imap(np.uint16, r.asDirect()[2])).astype(np.float32)
    return im

def load_depth2(path):
    d = scipy.misc.imread(path)
    d = d.astype(np.float32)
    return d

def save_depth(path, im):
    # PyPNG library is used since it allows to save 16-bit PNG
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    im_uint16 = np.round(im).astype(np.uint16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))

def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape((3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(info[eid]['cam_t_w2c']).reshape((3, 1))
    return info

def save_info(path, info):
    for im_id in sorted(info.keys()):
        im_info = info[im_id]
        if 'cam_K' in im_info.keys():
            im_info['cam_K'] = im_info['cam_K'].flatten().tolist()
        if 'cam_R_w2c' in im_info.keys():
            im_info['cam_R_w2c'] = im_info['cam_R_w2c'].flatten().tolist()
        if 'cam_t_w2c' in im_info.keys():
            im_info['cam_t_w2c'] = im_info['cam_t_w2c'].flatten().tolist()
    with open(path, 'w') as f:
        yaml.dump(info, f, Dumper=yaml.CDumper, width=10000)

def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts

def save_gt(path, gts):
    for im_id in sorted(gts.keys()):
        im_gts = gts[im_id]
        for gt in im_gts:
            if 'cam_R_m2c' in gt.keys():
                gt['cam_R_m2c'] = gt['cam_R_m2c'].flatten().tolist()
            if 'cam_t_m2c' in gt.keys():
                gt['cam_t_m2c'] = gt['cam_t_m2c'].flatten().tolist()
            if 'obj_bb' in gt.keys():
                gt['obj_bb'] = [int(x) for x in gt['obj_bb']]
    with open(path, 'w') as f:
        yaml.dump(gts, f, Dumper=yaml.CDumper, width=10000)

def load_results_sixd17(path):
    """
    Loads 6D object pose estimates from a file.

    :param path: Path to a file with poses.
    :return: List of the loaded poses.
    """
    with open(path, 'r') as f:
        res = yaml.load(f, Loader=yaml.CLoader)
        if not res['ests'] or res['ests'] == [{}]:
            res['ests'] = []
        else:
            for est in res['ests']:
                est['R'] = np.array(est['R']).reshape((3, 3))
                est['t'] = np.array(est['t']).reshape((3, 1))
                if isinstance(est['score'], basestring):
                    if 'nan' in est['score']:
                        est['score'] = 0.0
                    else:
                        raise ValueError('Bad type of score.')
    return res

def save_results_sixd17(path, res, run_time=-1):
    txt = 'run_time: ' + str(run_time) + '\n' # The first line contains run time
    txt += 'ests:\n'
    line_tpl = '- {{score: {:.8f}, ' \
                   'R: [' + ', '.join(['{:.8f}'] * 9) + '], ' \
                   't: [' + ', '.join(['{:.8f}'] * 3) + ']}}\n'
    for e in res['ests']:
        Rt = e['R'].flatten().tolist() + e['t'].flatten().tolist()
        txt += line_tpl.format(e['score'], *Rt)
    with open(path, 'w') as f:
        f.write(txt)

def load_errors(path):
    with open(path, 'r') as f:
        errors = yaml.load(f, Loader=yaml.CLoader)
    return errors

def save_errors(path, errors):
    with open(path, 'w') as f:
        line_tpl = '- {{im_id: {:d}, obj_id: {:d}, est_id: {:d}, ' \
                       'score: {:.8f}, errors: {}}}\n'
        error_tpl = '{:d}: {:.8f}'
        txt = ''
        for e in errors:
            txt_errors_elems = []
            for gt_id, error in e['errors'].items():
                txt_errors_elems.append(error_tpl.format(gt_id, error))
            txt_errors = '{' + ', '.join(txt_errors_elems) + '}'
            txt += line_tpl.format(e['im_id'], e['obj_id'], e['est_id'],
                                   e['score'], txt_errors)
        f.write(txt)

def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

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

    return model

def save_ply(path, pts, pts_colors=np.array([]), pts_normals=np.array([]), faces=np.array([])):
    """
    Saves a 3D mesh model to a PLY file.

    :param path: Path to the resulting PLY file.
    :param pts: nx3 ndarray
    :param pts_colors: nx3 ndarray
    :param pts_normals: nx3 ndarray
    :param faces: mx3 ndarray
    """
    pts_colors = np.array(pts_colors)
    if pts_colors.size != 0:
        assert(len(pts) == len(pts_colors))

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, 'w')
    f.write(
        'ply\n'
        'format ascii 1.0\n'
        #'format binary_little_endian 1.0\n'
        'element vertex ' + str(valid_pts_count) + '\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    if pts_normals.size != 0:
        f.write(
            'property float nx\n'
            'property float ny\n'
            'property float nz\n'
        )
    if pts_colors.size != 0:
        f.write(
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
        )
    if faces.size != 0:
        f.write(
            'element face ' + str(len(faces)) + '\n'
            'property list uchar int vertex_indices\n'
        )
    f.write('end_header\n')

    format_float = "{:.4f}"
    format_3float = " ".join((format_float for _ in range(3)))
    format_int = "{:d}"
    format_3int = " ".join((format_int for _ in range(3)))
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(format_3float.format(*pts[pt_id].astype(float)))
            if pts_normals.size != 0:
                f.write(' ')
                f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
            if pts_colors.size != 0:
                f.write(' ')
                f.write(format_3int.format(*pts_colors[pt_id].astype(int)))
            f.write('\n')
    for face in faces:
        f.write(' '.join(map(str, map(int, [len(face)] + list(face.squeeze())))) + ' ')
        f.write('\n')
    f.close()
