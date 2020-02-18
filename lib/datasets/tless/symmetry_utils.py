import numpy as np
from skimage.io import imread
from transforms3d.euler import mat2euler, euler2mat


def rectify_symmetric_rotation(R, rotation_groups):
    '''
    rectify pose according to https://arxiv.org/pdf/1908.07640.pdf Proposition 1
    :param R:                   [3,3]
    :param rotation_groups:     [n,3,3]
    :return:
    '''
    rg_inv = np.transpose(rotation_groups, [0, 2, 1])            # [n,3,3]
    rg_diff = R[None, :, :] @ rg_inv - np.identity(3)[None, :, :]  # [n,3,3]
    rg_diff_norm = np.linalg.norm(rg_diff.reshape([-1, 9]), 2, 1)
    rg_min_diff_idx = np.argmin(rg_diff_norm)
    return R @ rotation_groups[rg_min_diff_idx].T


def rectify_z_axis_symmetric_rotation(R):
    z, x, y = mat2euler(R, 'szyx')
    return euler2mat(0, x, y, 'szyx')


def TLESS_rotation_groups():
    return {
        1: gen_axis_group(60, 2),
        2: gen_axis_group(60, 2),
        # 3:gen_axis_group(45,2),
        # 4:
        # 5:gen_axis_group(180,2), #
        # 6:gen_axis_group(180,2),
        # 7:gen_axis_group(180,2), #
        8: gen_axis_group(180, 2),
        9: gen_axis_group(180, 2),
        # 10:gen_axis_group(180,2),
        # 11:gen_axis_group(180,2),
        # 12:gen_axis_group(180,2),
        # 13:gen_axis_group(180,2),
        # 14:gen_axis_group(180,2), # or z symmetric
        # 15:gen_axis_group(180,2), # or z symmetric
        # 16: gen_axis_group(180, 2),
        # 17:gen_axis_group(180,2), # or z symmetric
        19: gen_axis_group(180, 1),
        20: gen_axis_group(180, 1),
        # 21: gen_axis_group(180,1), #
        # 22: gen_axis_group(180,1),
        # 23: gen_axis_group(180,1),
        24: gen_axis_group(180, 2),
        # 25:
        # 26:
        27: gen_axis_group(90, 2),
        28: gen_axis_group(180, 2),
        29: gen_axis_group(180, 2),
        # 30: gen_axis_group(90, 2),
    }


def rectify_14(R):
    az,el,ti=mat2euler(R,'szyz')
    el = np.rad2deg(el)
    if el>-87: return rectify_symmetric_rotation(R,gen_axis_group(180,2))
    else: return rectify_z_axis_symmetric_rotation(R)


def rectify_15(R):
    az,el,ti=mat2euler(R,'szyz')
    el=np.rad2deg(el)
    if el>=-80:
        return R
    else:
        return rectify_z_axis_symmetric_rotation(R)


def rectify_16(R):
    az,el,ti=mat2euler(R,'szyz')
    el = np.rad2deg(el)
    if el>=-80 or el<=-101: return rectify_symmetric_rotation(R,gen_axis_group(180,2))
    else: return rectify_symmetric_rotation(R,gen_axis_group(90,2))


def rectify_17(R):
    az,el,ti=mat2euler(R,'szyz')
    el = np.rad2deg(el)
    if el >= -70:
        return R
    elif el<=-117:
        return rectify_symmetric_rotation(R,gen_axis_group(180,2))
    else:
        return rectify_z_axis_symmetric_rotation(R)


def rectify_30(R):
    az,el,ti=mat2euler(R,'szyz')
    el = np.rad2deg(el)
    # -80 81-159 161-
    if el>=-80: return rectify_symmetric_rotation(R,gen_axis_group(180,2))
    elif el>=-160: return rectify_symmetric_rotation(R,gen_axis_group(90,2))
    else: return rectify_z_axis_symmetric_rotation(R)


def TLESS_rectify(obj_id, R):
    TLESS_group = TLESS_rotation_groups()
    if obj_id in TLESS_group:
        return rectify_symmetric_rotation(R, TLESS_group[obj_id])
    elif obj_id in [3, 13]:
        return rectify_z_axis_symmetric_rotation(R)
    elif obj_id in [14, 15, 16, 17, 30]:
        return globals()[f'rectify_{obj_id}'](R)
    else:
        return R


def gen_axis_group(step, axis=2):
    rg = [np.identity(3)]
    for k in range(0, 360, step):
        angles = [0, 0, 0]
        angles[axis] = k
        rg.append(euler2mat(np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2]), 'sxyz'))
    rg = np.asarray(rg)
    return rg


# testing code below###############################
def validate_rectification_implementation(rg):
    for i in range(1024):
        angle = np.random.uniform(0, 2 * np.pi, 3)
        R = euler2mat(angle[0], angle[1], angle[2], 'sxyz')
        R0 = rectify_symmetric_rotation(R, rg)
        for k in range(1, rg.shape[0]):
            Rk = rectify_symmetric_rotation(R @ rg[k], rg)
            assert(np.mean(np.abs(R0 - Rk)) < 1e-10)


def validate_z_axis_symmetry_rectification_implementation():
    for i in range(1024):
        angle = np.random.uniform(0, 2 * np.pi, 4)
        R = euler2mat(angle[0], angle[1], angle[2], 'sxyz')
        Rs = euler2mat(angle[3], 0, 0, 'rzyz')
        R0 = rectify_z_axis_symmetric_rotation(R)
        Rr = rectify_z_axis_symmetric_rotation(R @ Rs)
        assert(np.mean(np.abs(R0 - Rr)) < 1e-10)


def render_data(obj_id=1, sample_num=5):
    import os
    make_bg_fn()
    x = np.full(2 * sample_num, 0)
    y = np.full(2 * sample_num, 0)
    z = np.full(2 * sample_num, 0.2)
    translation = np.stack([x, y, z], axis=1)
    euler = np.zeros([sample_num, 3], np.float32)
    euler[:, 1] = 30
    euler[:, 0] = np.random.uniform(0, 360, sample_num)

    if obj_id in [1, 2]:
        euler_sym = euler.copy()
        euler_sym[:, 0] += 60 * np.random.randint(0, 6)
    elif obj_id in [16, 28, 29]:
        euler_sym = euler.copy()
        euler_sym[:, 0] += 180
    elif obj_id in [27, 30]:
        euler_sym = euler.copy()
        euler_sym[:, 0] += 90 * np.random.randint(0, 4)
    elif obj_id in [19, 20]:
        euler[:, 0] = 90
        euler[:, 1] = 0
        euler[:, 2] += np.random.uniform(0, 360, euler.shape[0])
        euler_sym = euler.copy()
        euler_sym[:, 2] += 180
    elif obj_id in [14, 15, 17]:
        euler_sym = euler.copy()
        euler_sym[:, 0] += np.random.uniform(0, 360)
    else:
        raise NotImplementedError
    euler = np.concatenate([euler, euler_sym], 0)
    poses = np.concatenate([euler, translation], axis=1)
    np.save('data/tless/poses.npy', poses)

    blender = 'blender'
    blank_blend = 'lib/datasets/tless/blank.blend'
    backend = 'lib/datasets/tless/render_backend.py'
    ply_path = f'data/T-LESS/model_v2/models_cad/colobj_{obj_id:02}.ply'
    output_dir = 'data/render_sym_valid/'
    num_syn = sample_num * 2

    os.system('{} {} --background --python {} -- --cad_path {} --output_dir {} --num_syn {}'.
              format(blender, blank_blend, backend, ply_path, output_dir, num_syn))


def make_bg_fn():
    from skimage.io import imsave
    imsave('data/render_sym_valid/background.jpg', np.full([1024, 1024, 3], 127, np.uint8))


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from lib.utils.base_utils import read_pickle
    obj_id, sample_num = 17, 5
    render_data(obj_id, sample_num)
    for k in range(sample_num):
        RT0 = read_pickle(f'data/render_sym_valid/{k}_RT.pkl')['RT']
        RT1 = read_pickle(f'data/render_sym_valid/{k+5}_RT.pkl')['RT']
        R0 = TLESS_rectify(obj_id, RT0[:, :3])
        R1 = TLESS_rectify(obj_id, RT1[:, :3])
        assert(np.mean(np.abs(R0 - R1)) < 1e-6)
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.imshow(imread(f'data/render_sym_valid/{k}.png'))
        plt.subplot(122)
        plt.imshow(imread(f'data/render_sym_valid/{k+5}.png'))
        plt.show()

    validate_rectification_implementation(gen_axis_group(60, 2))
    validate_rectification_implementation(gen_axis_group(180, 2))
    validate_rectification_implementation(gen_axis_group(90, 2))
    validate_rectification_implementation(gen_axis_group(180, 1))
    validate_z_axis_symmetry_rectification_implementation()

    R1 = euler2mat(np.deg2rad(30), 0, 0, 'sxyz')
    R2 = euler2mat(0, np.deg2rad(15), 0, 'sxyz')
    R3 = euler2mat(0, 0, np.deg2rad(35), 'sxyz')
    R = euler2mat(np.deg2rad(30), np.deg2rad(15), np.deg2rad(35), 'sxyz')
    print(R3 @ R2 @ R1)
    print(R)
