import bpy
import os
import sys
import argparse
import numpy as np
import imp
import math
from transforms3d.euler import mat2euler
import pickle
import glob
import tqdm


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def setup_renderer(cfg):
    bpy.ops.object.select_all(action='TOGGLE')
    camera = bpy.data.objects['Camera']
    bpy.data.cameras['Camera'].clip_end = 10000

    # configure rendered image's parameters
    bpy.context.scene.render.resolution_x = cfg['width']
    bpy.context.scene.render.resolution_y = cfg['height']
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    cam_constraint = camera.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    parent_obj_to_camera(camera)

    bpy.context.scene.render.image_settings.file_format = cfg['format']

    return camera


def make_depth_node(cfg):
    bpy.data.scenes['Scene'].use_nodes = True
    bpy.data.scenes['Scene'].node_tree.links.clear()

    for n in bpy.data.scenes['Scene'].node_tree.nodes:
        bpy.data.scenes['Scene'].node_tree.nodes.remove(n)

    # render_node = bpy.data.scenes['Scene'].node_tree.nodes['Render Layers']
    render_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type="CompositorNodeRLayers")
    depth_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type="CompositorNodeOutputFile")
    map_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type="CompositorNodeMapRange")

    depth_node.base_path = ''
    depth_node.format.file_format = 'OPEN_EXR'
    depth_node.format.color_depth = '32'
    map_node.inputs[1].default_value = cfg['min_depth']
    map_node.inputs[2].default_value = cfg['max_depth']
    map_node.inputs[3].default_value = 0
    map_node.inputs[4].default_value = 1

    bpy.data.scenes['Scene'].node_tree.links.new(render_node.outputs['Depth'], map_node.inputs[0])
    bpy.data.scenes['Scene'].node_tree.links.new(map_node.outputs[0], depth_node.inputs[0])

    return depth_node


def build_plain_world():
    bpy.data.scenes['Scene'].cycles.film_transparent = True
    img_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type='CompositorNodeImage')
    scale_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type='CompositorNodeScale')
    alpha_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type='CompositorNodeAlphaOver')
    render_node = bpy.data.scenes['Scene'].node_tree.nodes['Render Layers']
    comp_node = bpy.data.scenes['Scene'].node_tree.nodes.new(type='CompositorNodeComposite')
    # comp_node = bpy.data.scenes['Scene'].node_tree.nodes['Composite']

    scale_node.space = 'RENDER_SIZE'
    comp_node.use_alpha = True
    bpy.data.scenes['Scene'].node_tree.links.new(img_node.outputs['Image'], scale_node.inputs['Image'])
    bpy.data.scenes['Scene'].node_tree.links.new(scale_node.outputs['Image'], alpha_node.inputs[1])
    bpy.data.scenes['Scene'].node_tree.links.new(render_node.outputs['Image'], alpha_node.inputs[2])
    bpy.data.scenes['Scene'].node_tree.links.new(alpha_node.outputs['Image'], comp_node.inputs['Image'])


def build_environment_world():
    bpy.data.worlds['World'].use_nodes = True
    env_node = bpy.data.worlds['World'].node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    back_node = bpy.data.worlds['World'].node_tree.nodes['Background']
    bpy.data.worlds['World'].node_tree.links.new(env_node.outputs['Color'], back_node.inputs['Color'])


def build_world(cfg):
    if cfg['world'] == 'plain':
        build_plain_world()
    elif cfg['world'] == 'environment':
        build_environment_world()
    else:
        raise KeyError('NO SUCH RENDER WORLD')


def make_plain_material(material):
    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Diffuse BSDF']
    mix_node = material.node_tree.nodes.new(type='ShaderNodeMixShader')
    gloss_node = material.node_tree.nodes.new(type='ShaderNodeBsdfGlossy')

    material.node_tree.links.new(mix_node.outputs['Shader'], mat_out.inputs['Surface'])
    material.node_tree.links.new(diffuse_node.outputs['BSDF'], mix_node.inputs[1])
    material.node_tree.links.new(gloss_node.outputs['BSDF'], mix_node.inputs[2])


def make_transparent_material(material):
    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Diffuse BSDF']
    gloss_node = material.node_tree.nodes.new(type='ShaderNodeBsdfGlossy')
    transp_node = material.node_tree.nodes.new(type='ShaderNodeBsdfTransparent')
    diffuse_gloss_mix_node = material.node_tree.nodes.new(type='ShaderNodeMixShader')
    tran_mix_node = material.node_tree.nodes.new(type='ShaderNodeMixShader')

    material.node_tree.links.new(tran_mix_node.outputs['Shader'], mat_out.inputs['Surface'])
    material.node_tree.links.new(diffuse_gloss_mix_node.outputs['Shader'], tran_mix_node.inputs[1])
    material.node_tree.links.new(transp_node.outputs['BSDF'], tran_mix_node.inputs[2])
    material.node_tree.links.new(diffuse_node.outputs['BSDF'], diffuse_gloss_mix_node.inputs[1])
    material.node_tree.links.new(gloss_node.outputs['BSDF'], diffuse_gloss_mix_node.inputs[2])


def set_material(material):
    nodes = material.node_tree.nodes
    nodes['Glossy BSDF'].inputs['Roughness'].default_value = np.random.uniform(0.05, 0.15)
    # nodes['Diffuse BSDF'].inputs['Color'].default_value = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), 0.3)


def add_shader_on_ply_object(obj):
    bpy.ops.material.new()
    material = list(bpy.data.materials)[0]

    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Diffuse BSDF']
    gloss_node = material.node_tree.nodes.new(type='ShaderNodeBsdfGlossy')
    attr_node = material.node_tree.nodes.new(type='ShaderNodeAttribute')

    material.node_tree.nodes.remove(diffuse_node)
    attr_node.attribute_name = 'Col'
    material.node_tree.links.new(attr_node.outputs['Color'], gloss_node.inputs['Color'])
    material.node_tree.links.new(gloss_node.outputs['BSDF'], mat_out.inputs['Surface'])

    gloss_node.inputs['Roughness'].default_value = 1

    obj.data.materials.append(material)

    return material


def set_camera(camera):
    camera.location = [0, 1.5, 0]
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = [np.pi / 2, 0, np.pi]
    bpy.context.scene.update()


def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return q1, q2, q3, q4


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return q1, q2, q3, q4


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return q1, q2, q3, q4


def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return q1, q2, q3, q4


def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return x, y, z


def obj_location(dist, azi, ele):
    ele = math.radians(ele)
    azi = math.radians(azi)
    x = dist * math.cos(azi) * math.cos(ele)
    y = dist * math.sin(azi) * math.cos(ele)
    z = dist * math.sin(ele)
    return x, y, z


def setup_light(scene):
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    azi = 0
    for i in range(0, 4):
        ele = np.random.uniform(0, 40)
        dist = np.random.uniform(1, 2)
        x, y, z = obj_location(dist, azi, ele)
        lamp_name = 'Lamp{}'.format(i)
        lamp_data = bpy.data.lamps.new(name=lamp_name, type='POINT')
        lamp_data.energy = np.random.uniform(0.5, 2)
        lamp = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)
        lamp.location = (x, y, z)
        scene.objects.link(lamp)
        azi += 90


class Renderer(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.blender_utils = imp.load_source('lib.utils.renderer.blender_utils', 'lib/utils/renderer/blender_utils.py')

        self.camera = setup_renderer(cfg)

        if cfg['cad_path'].endswith('.ply'):
            bpy.ops.import_mesh.ply(filepath=cfg['cad_path'])
        else:
            raise KeyError('No such render type')

        self.depth_node = make_depth_node(cfg)

        # set up the cycles render configuration
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.sample_clamp_indirect = 1.0
        bpy.context.scene.cycles.blur_glossy = 3.0
        bpy.context.scene.cycles.samples = 100

        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = 'GPU'

        for mesh in bpy.data.meshes:
            mesh.use_auto_smooth = True

        build_world(cfg)

        self.object = bpy.data.objects[os.path.basename(cfg['cad_path']).split('.')[0]]
        add_shader_on_ply_object(self.object)

    def transform_camera(self, pose):
        azi, ele, theta = pose[:3]
        cx, cy, cz = obj_centered_camera_pos(1, azi, ele)
        q1 = camPosToQuaternion(cx, cy, cz)
        q2 = camRotQuaternion(cx, cy, cz, theta)
        q = quaternionProduct(q2, q1)

        self.camera.location[0] = cx
        self.camera.location[1] = cy
        self.camera.location[2] = cz
        self.camera.rotation_mode = 'QUATERNION'
        self.camera.rotation_quaternion[0] = q[0]
        self.camera.rotation_quaternion[1] = q[1]
        self.camera.rotation_quaternion[2] = q[2]
        self.camera.rotation_quaternion[3] = q[3]
        bpy.context.scene.update()

        rotation_matrix = self.blender_utils.get_K_P_from_blender(self.camera)['RT'][:, :3]
        self.camera.location = -np.dot(rotation_matrix.T, pose[3:])
        bpy.context.scene.update()

    def save_pose(self, pose_path):
        K_P = self.blender_utils.get_K_P_from_blender(self.camera)
        save_pickle(K_P, pose_path)

    def _render(self, output_path):
        bpy.context.scene.render.filepath = output_path
        self.depth_node.file_slots[0].path = bpy.context.scene.render.filepath + '_depth.png'
        setup_light(bpy.context.scene)
        bpy.ops.render.render(write_still=True)

    def render(self, bg_img_path, pose, i):
        img_name = os.path.basename(bg_img_path)
        bpy.data.images.load(bg_img_path)

        if self.cfg['world'] == 'plain':
            bpy.data.scenes['Scene'].node_tree.nodes['Image'].image = bpy.data.images[img_name]
        elif self.cfg['world'] == 'environment':
            bpy.data.worlds['World'].node_tree.nodes['Environment Texture'].image = bpy.data.images[img_name]

        self.transform_camera(pose)
        self._render('{}/{}'.format(self.cfg['output_dir'], i))
        pose_path = '{}/{}_RT.pkl'.format(self.cfg['output_dir'], i)
        self.save_pose(pose_path)

        bpy.data.images.remove(bpy.data.images[img_name])

    def run(self):
        poses = np.load('data/tless/poses.npy')
        bg_imgs = np.load('data/tless/bg_imgs.npy')

        begin_i = len(os.listdir(self.cfg['output_dir'])) // 3
        for i in tqdm.tqdm(range(0, self.cfg['num_syn'])):
            i = 276
            bg_img = bg_imgs[np.random.randint(0, len(bg_imgs))]
            pose = poses[i]
            self.render(bg_img, pose, i)
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cad_path',
        type=str,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
    )
    parser.add_argument(
        '--num_syn',
        type=int,
    )

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    cfg = {
        'cad_path': args.cad_path,
        'output_dir': args.output_dir,
        'num_syn': args.num_syn,
        'width': 720,
        'height': 540,
        'format': 'PNG',
        'min_depth': 0,
        'max_depth': 10,
        'world': 'plain'
    }

    renderer = Renderer(cfg)
    renderer.run()
