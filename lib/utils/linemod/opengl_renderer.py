"""
opengl_render模块
=================

本模块主要用于加载ply模型文件,读取模型的elements及其properties信息.还可通过OpenGLRenderer.render方法合成rgb图或深度图(可用于合成数据集)
"""
# 标准库
import struct
# 第三方库
import numpy as np

class OpenGLRenderer(object):
    """
    OpenGLRenderer 管理ply模型文件的各项属性

    :param ply_path: ply文件路径
    :type ply_path: str
    """
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
        """
        __init__ 初始化函数

        :param ply_path: ply文件路径
        :type ply_path: str
        """
        self.ply_path = ply_path
        self.model = self.load_ply(self.ply_path)

    def load_ply(self, ply_path):
        """
        load_ply Loads a 3D mesh model from a PLY file.

        :param ply_path: 目标ply文件的路径
        :type ply_path: str
        :return: The loaded model given by a dictionary with items:
                 'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
                 'faces' (mx3 ndarray) - the latter three are optional.
        :rtype: dict('pts':narray, 'faces':narray, ...)
        """
        f = open(ply_path, 'r')
        """ply文件内的vertex数量"""
        n_pts = 0
        """ply文件内的face数量"""
        n_faces = 0
        """组成一个face的vertex数量"""
        face_n_corners = 3 # Only triangular faces are supported
        """vertex属性"""
        pt_props = []
        """face属性"""
        face_props = []
        """当前打开的ply文件是否为二进制编码"""
        is_binary = False
        """当前是否正在处理ply头部vertex信息"""
        header_vertex_section = False
        """当前是否正在处理ply头部face信息"""
        header_face_section = False

        # Read header
        while True:
            line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
            if line.startswith('element vertex'): # 点元素头部信息起始行
                n_pts = int(line.split()[-1])     # 读取ply文件中vertex的个数
                header_vertex_section = True
                header_face_section = False
            elif line.startswith('element face'): # 面元素头部信息起始行
                n_faces = int(line.split()[-1])   # 读取ply文件中face个数
                header_vertex_section = False
                header_face_section = True
            elif line.startswith('element'): # Some other element
                header_vertex_section = False
                header_face_section = False
            elif line.startswith('property') and header_vertex_section: # vertex属性
                # (name of the property, data type)
                pt_props.append((line.split()[-1], line.split()[-2]))
            elif line.startswith('property list') and header_face_section:  # 只能读取组成face顶点的列表属性,不支持读取其它类型的face属性
                elems = line.split()
                if elems[-1] == 'vertex_indices':
                    # (name of the property, data type)
                    face_props.append(('n_corners', elems[2]))
                    for i in range(face_n_corners):
                        face_props.append(('ind_' + str(i), elems[3]))
                else:
                    print('Warning: Not supported face property: ' + elems[-1])
            elif line.startswith('format'): # 文件的编码格式
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('end_header'): # 头部信息结束标志行
                break

        # Prepare data structures
        model = {}
        model['pts'] = np.zeros((n_pts, 3), np.float)
        if n_faces > 0:
            model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

        """vertex的属性名称"""
        pt_props_names = [p[0] for p in pt_props]
        """是否存在vertex的归一化坐标信息"""
        is_normal = False
        if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
            is_normal = True
            model['normals'] = np.zeros((n_pts, 3), np.float)

        """是否存在vertex的颜色信息"""
        is_color = False
        if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
            is_color = True
            model['colors'] = np.zeros((n_pts, 3), np.float)

        """是否存在vertex的纹理信息"""
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
        for pt_id in range(n_pts):  # 每次读取一个point,即一行的数据
            """保存属性值{prop_name:prop_val}"""
            prop_vals = {}
            """可保存的属性"""
            load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                          'red', 'green', 'blue', 'texture_u', 'texture_v']
            if is_binary:   # 若为二进制文件,则基于struct模块,按指定的数据类型读取
                for prop in pt_props:
                    format = formats[prop[1]]
                    val = struct.unpack(format[0], f.read(format[1]))[0]
                    if prop[0] in load_props:   # 只能读取限定的属性
                        prop_vals[prop[0]] = val
            else:           # 若为txt文件,按行读取即可
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(pt_props):
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = elems[prop_id]

            # 将当前vertex的x, y, z坐标保存至model内
            model['pts'][pt_id, 0] = float(prop_vals['x'])
            model['pts'][pt_id, 1] = float(prop_vals['y'])
            model['pts'][pt_id, 2] = float(prop_vals['z'])

            if is_normal:  # 若存在vertex归一化坐标属性,则保存
                model['normals'][pt_id, 0] = float(prop_vals['nx'])
                model['normals'][pt_id, 1] = float(prop_vals['ny'])
                model['normals'][pt_id, 2] = float(prop_vals['nz'])

            if is_color:  # 若存在vertex颜色属性,则保存
                model['colors'][pt_id, 0] = float(prop_vals['red'])
                model['colors'][pt_id, 1] = float(prop_vals['green'])
                model['colors'][pt_id, 2] = float(prop_vals['blue'])

            if is_texture:  # 若存在vertex纹理属性,则保存
                model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
                model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

        # Load faces
        for face_id in range(n_faces):  # 每次读取一个face属性,即一行的数据
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
            
            # 将读取的face属性保存至model内
            model['faces'][face_id, 0] = int(prop_vals['ind_0'])
            model['faces'][face_id, 1] = int(prop_vals['ind_1'])
            model['faces'][face_id, 2] = int(prop_vals['ind_2'])

        f.close()
        model['pts'] *= 1000.

        return model

    def render(self, pose, K, img_size, render_type='depth'):
        """
        render 给定位姿pose,相机内参K,图像尺寸img_size,渲染出一副合成的rgb图像或深度图

        :param pose: 模型位姿
        :type pose: narray
        :param K: 相机内参
        :type K: narray
        :param img_size: 图像尺寸(宽*高)
        :type img_size: narray
        :param render_type: 合成rgb图像("rgb")或深度图(depth), 默认值为'depth'
        :type render_type: str
        :return: 合成的rgb图像或深度图
        :rtype: narray
        """
        from .opengl_backend import render
        R = pose[:, :3]
        t = pose[:, 3:] * 1000.
        model = self.model
        if render_type == 'depth':
            return render(model, im_size=img_size, K=K, R=R, t=t, clip_near=10, clip_far=10000, mode='depth') / 1000.
        else:
            return render(model, im_size=img_size, K=K, R=R, t=t, clip_near=10, clip_far=10000, mode='rgb')

