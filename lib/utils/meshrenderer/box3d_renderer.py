# -*- coding: utf-8 -*-
import os
import numpy as np

from OpenGL.GL import *

import gl_utils as gu

class Renderer(object):

    MAX_FBO_WIDTH = 2000
    MAX_FBO_HEIGHT = 2000

    def __init__(self, models_cad_files, samples, W, H, vertex_tmp_store_folder='.', debug_mode=False):
        self.W, self.H = W, H
        self._context = gu.OffscreenContext()

        self._samples = 1

        self._rgb_tex = gu.Texture(GL_TEXTURE_2D, 1, GL_RGB32F, self.W, self.H)
        self._rgb_tex.setFilter(GL_NEAREST, GL_NEAREST)
        self._rgb_tex.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE)

        self._edge_tex = gu.Texture(GL_TEXTURE_2D, 1, GL_RGB32F, self.W, self.H)
        self._edge_tex.setFilter(GL_NEAREST, GL_NEAREST)
        self._edge_tex.setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE)

        # TEXTURE BUFFER
        handles = [self._rgb_tex.makeResident(), self._edge_tex.makeResident()]
        self._material_buffer = gu.ShaderStorage(2, np.array(handles, dtype=np.uint64), False)
        self._material_buffer.bind()

        self._fbo = gu.Framebuffer( {   GL_COLOR_ATTACHMENT0: self._rgb_tex,
                                        GL_DEPTH_ATTACHMENT: gu.Renderbuffer(GL_DEPTH_COMPONENT24, self.W, self.H)}  )

        # FOR GRADIENTS
        if debug_mode:
            self._gradient_fbo = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_R32F, self.W, self.H),
                                                   GL_COLOR_ATTACHMENT1: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB32F, self.W, self.H),
                                                   GL_COLOR_ATTACHMENT2: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB32F, self.W, self.H) } )
            glNamedFramebufferDrawBuffers(self._gradient_fbo.id, 3, np.array( (GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2),dtype=np.uint32 ) )
        else:
            self._gradient_fbo = gu.Framebuffer( { GL_COLOR_ATTACHMENT0: gu.Texture(GL_TEXTURE_2D, 1, GL_RGB32F, self.W, self.H) } )
        # VAO
        vert_norms = gu.geo.load_meshes(models_cad_files, vertex_tmp_store_folder, recalculate_normals=True )
        
        vertices = np.empty(0, dtype=np.float32)
        self.min_vert = {}
        self.max_vert = {}
        for obj_id,vert_norm in enumerate(vert_norms):
            vertices = np.hstack((vertices, np.hstack((vert_norm[0], vert_norm[1])).reshape(-1)))
            # if obj==obj_id:
            self.min_vert[obj_id] = np.min(vert_norm[0],axis=0)
            self.max_vert[obj_id] = np.max(vert_norm[0],axis=0)

        print  self.min_vert,  self.max_vert

        vao = gu.VAO({(gu.Vertexbuffer(vertices), 0, 6*4):
                        [   (0, 3, GL_FLOAT, GL_FALSE, 0*4),
                            (1, 3, GL_FLOAT, GL_FALSE, 3*4)]})
        vao.bind()

        # IBO
        sizes = [vert[0].shape[0] for vert in vert_norms]
        offsets = [sum(sizes[:i]) for i in xrange(len(sizes))]

        ibo = gu.IBO(sizes, np.ones(len(vert_norms)), offsets, np.zeros(len(vert_norms)))
        ibo.bind()

        gu.Shader.shader_folder = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'shader')
        # self.scene_shader = gu.Shader('common_shader.vs', 'common_shader.frag')
        # self.scene_shader.compile()

        # self.edge_shader = gu.Shader('screen.vs', 'edge_shader.frag')
        # self.edge_shader.compile_and_use()
        # glUniform1i(1, 1 if debug_mode else 0)

        # self.outline_shader = gu.Shader('screen.vs', 'edge_shader_lineout.frag')
        # self.outline_shader.compile()

        # self.screen_shader = gu.Shader('screen.vs', 'screen.frag')
        # self.screen_shader.compile()

        self.line_shader = gu.Shader('line.vs', 'line.frag')
        self.line_shader.compile()

        self._scene_buffer = gu.ShaderStorage(0, gu.Camera().data , True)
        self._scene_buffer.bind()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.camera = gu.Camera()
        self.debug_mode = debug_mode
        glLineWidth(3)

    def upload_edges(self, scale, pixels):
        H, W = pixels.shape[:2]
        pixels = np.ascontiguousarray(np.flipud(pixels))
        self._edge_tex.subImage(0, 0, 0, W, H, GL_RG, GL_FLOAT, pixels)

    def render(self, obj_id, K, R, t, near, far, row=0.0, col=0.0, reconst=False):
        W, H = self.W, self.H
        
        camera = gu.Camera()
        camera.realCamera(W, H, K, R, t, near, far)
        self._scene_buffer.update(camera.data)
        
        self._fbo.bind()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, W, H)

        # if reconst:
        #     self.scene_shader.use()
        #     glDrawArraysIndirect(GL_TRIANGLES, ctypes.c_void_p(obj_id*16))
        
        #     self._gradient_fbo.bind()
        #     self.outline_shader.use()
        #     glClear(GL_COLOR_BUFFER_BIT)
        #     glViewport(0, 0, W, H)
        #     glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

        self.line_shader.use()
        #number of lines
        glUniform3f(0, self.min_vert[obj_id][0], self.min_vert[obj_id][1], self.min_vert[obj_id][2])
        glUniform3f(1, self.max_vert[obj_id][0], self.max_vert[obj_id][1], self.max_vert[obj_id][2])
        glDrawArraysInstanced(GL_LINES, 0, 2, 15)

        rgb_flipped = np.frombuffer( glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE), dtype=np.uint8 ).reshape(H,W,3)
        return np.flipud(rgb_flipped).copy()
 
    # def render(self, obj_id, K, R, t, near, far, scale, row, col, outline=False,reconst=False):

        # assert scale <= 1.0
        # W = int(1.*self.W*scale)
        # H = int(1.*self.H*scale)
        # K[[0,1,2],[0,1,2]] *= scale
        
        # self._fbo.bind()
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glViewport(0, 0, W, H)

        # self.camera.real_camera(W, H, K, R, t, near, far)
        # self._scene_buffer.update(self.camera.data)

        # self.scene_shader.use()
        
        # glEnable(GL_DEPTH_TEST)
        # if reconst:
        #     glDrawArraysIndirect(GL_TRIANGLES, ctypes.c_void_p(obj_id*16))
        # glDisable(GL_DEPTH_TEST)

        # self._gradient_fbo.bind()
        # glClear(GL_COLOR_BUFFER_BIT)
        # glViewport(0, 0, W, H)

        # if outline:
        #     self.outline_shader.use()
        # else:
        #     self.edge_shader.use()
        # glUniform1f(0, scale)
        # glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        
        # self.line_shader.use()
        # #number of lines
        # glUniform3f(0, self.min_vert[0], self.min_vert[1], self.min_vert[2])
        # glUniform3f(1, self.max_vert[0], self.max_vert[1], self.max_vert[2])
        # glDrawArraysInstanced(GL_LINES, 0, 2, 15)

        # #rg_flipped = glReadPixels(x0, self.H-y0, w, h, GL_RGB, GL_FLOAT).reshape(h, w, 3)
        # if outline:
        #     rg_scene_flipped = glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT).reshape(H, W, 3)
        #     rg_scene = np.flipud(rg_scene_flipped).copy()
        #     return (rg_scene,)
        # else:
        #     if not self.debug_mode:
        #         rg_scene_flipped = glReadPixels(0, 0, W, H, GL_RED, GL_FLOAT).reshape(H, W)
        #         rg_scene = np.flipud(rg_scene_flipped).copy()
        #         return (rg_scene,)
        #     else:
        #         glNamedFramebufferReadBuffer(self._gradient_fbo.id, GL_COLOR_ATTACHMENT0)
        #         grad_similarity_flipped = glReadPixels(0, 0, W, H, GL_RED, GL_FLOAT).reshape(H, W, 1)
        #         grad_similarity = np.flipud(grad_similarity_flipped).copy()

        #         glNamedFramebufferReadBuffer(self._gradient_fbo.id, GL_COLOR_ATTACHMENT1)
        #         canny_flipped = glReadPixels(0, 0, W, H, GL_BGR, GL_FLOAT).reshape(H, W, 3)
        #         canny = np.flipud(canny_flipped).copy()

        #         glNamedFramebufferReadBuffer(self._gradient_fbo.id, GL_COLOR_ATTACHMENT2)
        #         color_3d_edges_flipped = glReadPixels(0, 0, W, H, GL_BGR, GL_FLOAT).reshape(H, W, 3)
        #         color_3d_edges = np.flipud(color_3d_edges_flipped).copy()

        #         return grad_similarity, canny, color_3d_edges


    def render_color_many(self, obj_ids, W, H, K, Rs, ts, near, far, rows, cols):
        assert W <= Renderer.MAX_FBO_WIDTH and H <= Renderer.MAX_FBO_HEIGHT

        if self._samples > 1:
            self._render_fbo.bind()

        self._fbo.bind()
        self.scene_shader.use()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, W, H)

        for obj_id, R, t, row, col in zip(obj_ids, Rs, ts, rows, cols):
            camera = gu.Camera()
            camera.realCamera(W, H, K.copy(), R, t, near, far)
            #camera.real_camera(W, H, K.copy(), R, t, near, far, r=row, c=col)
            self._scene_buffer.update(camera.data)

            #glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, ctypes.c_void_p(obj_id*4*5))
            glDrawArraysIndirect(GL_TRIANGLES, ctypes.c_void_p(obj_id*16))

        self._gradient_fbo.bind()
        glClear(GL_COLOR_BUFFER_BIT)
        glViewport(0, 0, W, H)
        self.outline_shader.use()

        if self._samples > 1:
            for i in xrange(2):
                glNamedFramebufferReadBuffer(self._render_fbo.id, GL_COLOR_ATTACHMENT0 + i)
                glNamedFramebufferDrawBuffer(self._fbo.id, GL_COLOR_ATTACHMENT0 + i)
                glBlitNamedFramebuffer(self._render_fbo.id, self._fbo.id, 0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST)
            self._fbo.bind()

        glUniform1f(0, 1.0)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

        self.line_shader.use()
        for obj_id, R, t, row, col in zip(obj_ids, Rs, ts, rows, cols):
            camera = gu.Camera()
            camera.realCamera(W, H, K.copy(), R, t, near, far)
            #camera.real_camera(W, H, K.copy(), R, t, near, far, r=row, c=col)
            self._scene_buffer.update(camera.data)

            glDrawArraysInstanced(GL_LINES, 0, 2, 3)

        rg_scene_flipped = glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT).reshape(H, W, 3)
        rg_scene = np.flipud(rg_scene_flipped).copy()

        return (rg_scene, None)

    def close(self):
        self._context.close()