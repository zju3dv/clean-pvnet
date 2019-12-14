# -*- coding: utf-8 -*-
import numpy as np

from OpenGL.GL import *
from OpenGL.GL.NV.bindless_texture import *

class Texture(object):

    def __init__(self, tex_type, levels, internalformat, W, H):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(tex_type, len(self.__id), self.__id)
        glTextureStorage2D(self.__id[0], levels, internalformat, W, H)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t, wrap_r=None):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)
        if wrap_r!=None:
            glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_R, wrap_r)

    def subImage(self, level, xoffset, yoffset, width, height,
                        data_format, data_type, pixels):
        glTextureSubImage2D(self.__id[0], level, xoffset, yoffset, 
            width, height, data_format, data_type, pixels)

    def generate_mipmap(self):
        glGenerateTextureMipmap(self.__id[0])

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]

class Texture1D(object):

    def __init__(self, levels, internalformat, W):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(GL_TEXTURE_1D, len(self.__id), self.__id)
        glTextureStorage1D(self.__id[0], levels, internalformat, W)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t, wrap_r=None):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)
        if wrap_r!=None:
            glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_R, wrap_r)

    def subImage(self, level, xoffset, width, 
                        data_format, data_type, pixels):
        glTextureSubImage1D(self.__id[0], level, xoffset,  
            width, data_format, data_type, pixels)

    def generate_mipmap(self):
        glGenerateTextureMipmap(self.__id[0])

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]

class Texture3D(object):

    def __init__(self, tex_type, levels, internalformat, W, H, C):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(tex_type, len(self.__id), self.__id)
        glTextureStorage3D(self.__id[0], levels, internalformat, W, H, C)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t, wrap_r=None):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)
        if wrap_r!=None:
            glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_R, wrap_r)

    def subImage(self, level, xoffset, yoffset, zoffset, width, height, depth,
                        data_format, data_type, pixels):
        glTextureSubImage3D(self.__id[0], level, xoffset, yoffset, zoffset,
            width, height, depth, data_format, data_type, pixels)

    def generate_mipmap(self):
        glGenerateTextureMipmap(self.__id[0])

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]

class TextureMultisample(object):

    def __init__(self, samples, internalformat, W, H, fixedsamplelocations):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, len(self.__id), self.__id)
        glTextureStorage2DMultisample(self.__id[0], samples, internalformat, W, H, fixedsamplelocations)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)

    def subImage(self, level, xoffset, yoffset, width, height,
                        data_format, data_type, pixels):
        glTextureSubImage2D(self.__id[0], level, xoffset, yoffset, 
            width, height, data_format, data_type, pixels)

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]
