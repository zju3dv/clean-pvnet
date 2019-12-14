# -*- coding: utf-8 -*-
import logging as log
import os

'''
if not os.environ.get( 'PYOPENGL_PLATFORM' ):
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
'''

from OpenGL.GL import *
from OpenGL.EGL import *

from OpenGL.GL.NV.bindless_texture import *

class OffscreenContext(object):

    def __init__(self):
        major, minor = ctypes.c_long(), ctypes.c_long()
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        log.info( 'Display return value: %s', display)
        log.info( 'Display address: %s', display.address)
        if not eglInitialize(display, major, minor):
            raise RuntimeError('Unable to initialize')
        log.info('EGL version %s.%s', major.value, minor.value)
        
        num_configs = ctypes.c_long()
        configs = (EGLConfig*2)()
        
        eglChooseConfig(display, None, configs, 2, num_configs)
        
        eglBindAPI(EGL_OPENGL_API)

        ctx = eglCreateContext(display, configs[0], EGL_NO_CONTEXT, None)
        if ctx == EGL_NO_CONTEXT:
            raise RuntimeError( 'Unable to create context' )

        eglMakeCurrent( display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx )

        if not glInitBindlessTextureNV():
            raise RuntimeError('Bindless Textures not supported')

        self.__display = display

    def close(self):
        eglTerminate(self.__display)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()