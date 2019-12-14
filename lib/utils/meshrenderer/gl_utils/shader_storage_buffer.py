# -*- coding: utf-8 -*-
import numpy as np

from OpenGL.GL import *

class ShaderStorage(object):

	def __init__(self, biding_point, data, dynamic=True):
		self.__dynamic = dynamic
		self.__binding_point = biding_point
		self.__id = np.empty(1, dtype=np.uint32)
		glCreateBuffers(1, self.__id)
		code = 0 if not dynamic else GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT| GL_MAP_PERSISTENT_BIT
		glNamedBufferStorage( self.__id[0], data.nbytes, data, code)
		self.__data_size = data.nbytes

	def bind(self):
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, self.__binding_point, self.__id[0], 0, self.__data_size)

	def update(self, data, offset=0, nbytes=None):
		nbytes = data.nbytes if nbytes == None else nbytes
		assert self.__dynamic == True, 'Updating of a non-updatable buffer.'
		assert data.nbytes == self.__data_size, 'Please put the same amount of data into the buffer as during creation.'
		glNamedBufferSubData(self.__id[0], offset, nbytes, data)