# -*- coding: utf-8 -*-
import numpy as np

from OpenGL.GL import *

from lib.utils.meshrenderer.gl_utils.ebo import EBO

class VAO(object):

    def __init__(self, vbo_attrib, ebo=None):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateVertexArrays(len(self.__id), self.__id)
        i = 0
        for vbo_offset_stride, attribs in vbo_attrib.items():
            vbo = vbo_offset_stride[0]
            offset = vbo_offset_stride[1]
            stride = vbo_offset_stride[2]
            for attrib in attribs:
                attribindex = attrib[0]
                size = attrib[1]
                attribtype = attrib[2]
                normalized = attrib[3]
                relativeoffset = attrib[4]
                glVertexArrayAttribFormat(self.__id, attribindex, size, attribtype, normalized, relativeoffset)
                glVertexArrayAttribBinding(self.__id, attribindex, i)
                glEnableVertexArrayAttrib(self.__id, attribindex)
            glVertexArrayVertexBuffer(self.__id, i, vbo.id, offset, stride)
            i += 1
        if ebo != None:
            if isinstance(ebo, EBO):
                glVertexArrayElementBuffer(self.__id, ebo.id)
            else:
                ValueError('Invalid EBO type.')

    def bind(self):
        glBindVertexArray(self.__id)



