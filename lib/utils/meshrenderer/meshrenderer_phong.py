# -*- coding: utf-8 -*-
import os
import numpy as np

from OpenGL.GL import *

import lib.utils.meshrenderer.gl_utils as gu

from .pysixd import misc

class Renderer(object):

    MAX_FBO_WIDTH = 2000
    MAX_FBO_HEIGHT = 2000

    def __init__(self, models_cad_files, samples=1, vertex_tmp_store_folder='.',clamp=False, vertex_scale=1.0):
        self._samples = samples
        self._context = gu.OffscreenContext()

        # FBO
        W, H = Renderer.MAX_FBO_WIDTH, Renderer.MAX_FBO_HEIGHT
        self._fbo = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB8, W, H),
                                      GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, W, H),
                                      GL_DEPTH_ATTACHMENT: gu.Renderbuffer(GL_DEPTH_COMPONENT32F, W, H) } )

        self._fbo_depth = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB8, W, H),
                                      GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, W, H),
                                      GL_DEPTH_ATTACHMENT: gu.Renderbuffer(GL_DEPTH_COMPONENT32F, W, H) } )
        glNamedFramebufferDrawBuffers(self._fbo.id, 2, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )
        glNamedFramebufferDrawBuffers(self._fbo_depth.id, 2, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )

        if self._samples > 1:
            self._render_fbo = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.TextureMultisample(self._samples, GL_RGB8, W, H, True),
                                                 GL_COLOR_ATTACHMENT1: gu.TextureMultisample(self._samples, GL_R32F, W, H, True),
                                                 GL_DEPTH_STENCIL_ATTACHMENT: gu.RenderbufferMultisample(self._samples, GL_DEPTH32F_STENCIL8, W, H) } )
            glNamedFramebufferDrawBuffers(self._render_fbo.id, 2, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )

        self._fbo.bind()

        # VAO
        attributes = gu.geo.load_meshes_sixd(models_cad_files, vertex_tmp_store_folder, recalculate_normals=False)

        vertices = []
        indices = []
        for vertex, normal, color, faces in attributes:
            indices.append( faces.flatten() )
            vertices.append(np.hstack((vertex * vertex_scale, normal, color/255.0)).flatten())

        indices = np.hstack(indices).astype(np.uint32)
        vertices = np.hstack(vertices).astype(np.float32)

        vao = gu.VAO({(gu.Vertexbuffer(vertices), 0, 9*4):
                        [   (0, 3, GL_FLOAT, GL_FALSE, 0*4),
                            (1, 3, GL_FLOAT, GL_FALSE, 3*4),
                            (2, 3, GL_FLOAT, GL_FALSE, 6*4)]}, gu.EBO(indices))
        vao.bind()

        # IBO
        vertex_count = [np.prod(vert[3].shape) for vert in attributes]
        instance_count = np.ones(len(attributes))
        first_index = [sum(vertex_count[:i]) for i in range(len(vertex_count))]
        
        vertex_sizes = [vert[0].shape[0] for vert in attributes]
        base_vertex = [sum(vertex_sizes[:i]) for i in range(len(vertex_sizes))]
        base_instance = np.zeros(len(attributes))

        ibo = gu.IBO(vertex_count, instance_count, first_index, base_vertex, base_instance)
        ibo.bind()

        gu.Shader.shader_folder = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'shader')
        # if clamp:
        #     shader = gu.Shader('depth_shader_phong.vs', 'depth_shader_phong_clamped.frag')
        # else:
        shader = gu.Shader('depth_shader_phong.vs', 'depth_shader_phong.frag')
        shader.compile_and_use()

        self._scene_buffer = gu.ShaderStorage(0, gu.Camera().data , True)
        self._scene_buffer.bind()

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def set_light_pose(self, direction):
        glUniform3f(1, direction[0], direction[1], direction[2])

    def set_ambient_light(self, a):
        glUniform1f(0, a)

    def set_diffuse_light(self, a):
        glUniform1f(2, a)

    def set_specular_light(self, a):
        glUniform1f(3, a)


    def render(self, obj_id, W, H, K, R, t, near, far, random_light=False, phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3}):
        assert W <= Renderer.MAX_FBO_WIDTH and H <= Renderer.MAX_FBO_HEIGHT

        if self._samples > 1:
            self._render_fbo.bind()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |  GL_STENCIL_BUFFER_BIT)
        glViewport(0, 0, W, H)

        camera = gu.Camera()
        camera.realCamera(W, H, K, R, t, near, far)

        self._scene_buffer.update(camera.data)
        # print phong
        if random_light:
            self.set_light_pose( 1000.*np.random.random(3) )
            self.set_ambient_light(phong['ambient'])
            self.set_diffuse_light(phong['diffuse'] + 0.1*(2*np.random.rand()-1))
            self.set_specular_light(phong['specular'] + 0.1*(2*np.random.rand()-1))
            # self.set_ambient_light(phong['ambient'])
            # self.set_diffuse_light(0.7)
            # self.set_specular_light(0.3)
        else:
            self.set_light_pose( np.array([400., 400., 400]) )
            self.set_ambient_light(phong['ambient'])
            self.set_diffuse_light(phong['diffuse'])
            self.set_specular_light(phong['specular'])


        # self.set_ambient_light(0.4)
        # self.set_diffuse_light(0.7)

        # self.set_ambient_light(0.5)
        # self.set_diffuse_light(0.4)
        # self.set_specular_light(0.1)

        glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(obj_id*4*5))

        # if self._samples > 1:
        #     for i in xrange(2):
        #         glNamedFramebufferReadBuffer(self._render_fbo.id, GL_COLOR_ATTACHMENT0 + i)
        #         glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0 + i)
        #         glBlitNamedFramebuffer(self._render_fbo.id, self._fbo.id, 0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        #     self._fbo.bind()

        if self._samples > 1:
            self._fbo.bind()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(obj_id*4*5))

            glNamedFramebufferReadBuffer(self._render_fbo.id, GL_COLOR_ATTACHMENT0)
            glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
            glBlitNamedFramebuffer(self._render_fbo.id, self._fbo.id, 0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST)

            glNamedFramebufferDrawBuffers(self._fbo.id, 2, np.array( (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1),dtype=np.uint32 ) )

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
        bgr_flipped = np.frombuffer( glReadPixels(0, 0, W, H, GL_BGR, GL_UNSIGNED_BYTE), dtype=np.uint8 ).reshape(H,W,3)
        bgr = np.flipud(bgr_flipped).copy()

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
        depth_flipped = glReadPixels(0, 0, W, H, GL_RED, GL_FLOAT).reshape(H,W)
        depth = np.flipud(depth_flipped).copy()

        return bgr, depth

    def render_many(self, obj_ids, W, H, K, Rs, ts, near, far, random_light=True, phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3}):
        assert W <= Renderer.MAX_FBO_WIDTH and H <= Renderer.MAX_FBO_HEIGHT

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, W, H)

        if random_light:
            self.set_light_pose( 1000.*np.random.random(3) )
            self.set_ambient_light(phong['ambient'] + 0.1*(2*np.random.rand()-1))
            self.set_diffuse_light(phong['diffuse'] + 0.1*(2*np.random.rand()-1))
            self.set_specular_light(phong['specular'] + 0.1*(2*np.random.rand()-1))
            # self.set_ambient_light(phong['ambient'])
            # self.set_diffuse_light(0.7)
            # self.set_specular_light(0.3)
        else:
            self.set_light_pose( np.array([400., 400., 400]) )
            self.set_ambient_light(phong['ambient'])
            self.set_diffuse_light(phong['diffuse'])
            self.set_specular_light(phong['specular'])

        bbs = []
        for i in range(len(obj_ids)):
            o = obj_ids[i]
            R = Rs[i]
            t = ts[i]
            camera = gu.Camera()
            camera.realCamera(W, H, K, R, t, near, far)
            self._scene_buffer.update(camera.data)

            self._fbo.bind()
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(o*4*5))

            self._fbo_depth.bind()
            glViewport(0, 0, W, H)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(o*4*5))

            glNamedFramebufferReadBuffer(self._fbo_depth.id, GL_COLOR_ATTACHMENT1)
            depth_flipped = glReadPixels(0, 0, W, H, GL_RED, GL_FLOAT).reshape(H,W)
            depth = np.flipud(depth_flipped).copy()

            ys, xs = np.nonzero(depth > 0)
            obj_bb = misc.calc_2d_bbox(xs, ys, (W,H))
            bbs.append(obj_bb)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo.id)
        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0)
        bgr_flipped = np.frombuffer( glReadPixels(0, 0, W, H, GL_BGR, GL_UNSIGNED_BYTE), dtype=np.uint8 ).reshape(H,W,3)
        bgr = np.flipud(bgr_flipped).copy()

        glNamedFramebufferReadBuffer(self._fbo.id, GL_COLOR_ATTACHMENT1)
        depth_flipped = glReadPixels(0, 0, W, H, GL_RED, GL_FLOAT).reshape(H,W)
        depth = np.flipud(depth_flipped).copy()

        return bgr, depth, bbs
    def close(self):
        self._context.close()