# -*- coding: utf-8 -*-
import numpy as np

from OpenGL.GL import *

class Vertexbuffer(object):

    def __init__(self, data, dynamic=False):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateBuffers(len(self.__id), self.__id)
        code = 0 if not dynamic else GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT| GL_MAP_PERSISTENT_BIT
        glNamedBufferStorage(self.__id, data.nbytes, data, code)

    @property
    def id(self):
        return self.__id