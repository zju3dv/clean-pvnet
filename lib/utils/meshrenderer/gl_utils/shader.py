# -*- coding: UTF-8 -*-
import logging as log
import os

from OpenGL.GL import *

class Shader(object):

    shader_folder = None
    active_shader = None

    def __init__(self, *shaderPaths):
        self.__shader = []
        endings = [ s[s.rindex('.')+1:] for s in shaderPaths]
        for end, shType in zip(['vs', 'tcs', 'tes', 'gs', 'frag', 'cs'],
                        [GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHADER]):
            try:
                shader = shaderPaths[endings.index(end)]
                self.__shader.append((shader,shType))
            except ValueError as e:
                pass

    def compile(self, varyings=None):
        log.debug('Compiling shader.')
        shaderIDs = []
        for shader in self.__shader:
            path = None
            if Shader.shader_folder != None:
                path = os.path.join(Shader.shader_folder, shader[0])
            else:
                path = shader[0]
            code = Shader.__readFile__(path)
            shaderID = Shader.__createShader__(path, shader[1], code)
            shaderIDs.append(shaderID)

        self.__program = glCreateProgram()
        for shaderID in shaderIDs:
            glAttachShader(self.__program, shaderID)

        if varyings is not None:            
            LP_c_char = ctypes.POINTER(ctypes.c_char)

            argv = (LP_c_char * len(varyings))()
            for i, arg in enumerate(varyings):
                enc_arg = arg.encode('utf-8')
                argv[i] = ctypes.create_string_buffer(enc_arg) 

            glTransformFeedbackVaryings(self.__program, 2, argv, GL_SEPARATE_ATTRIBS)

        glLinkProgram(self.__program)
        if not glGetProgramiv(self.__program, GL_LINK_STATUS):
            log.error(glGetProgramInfoLog(self.__program))
            raise RuntimeError('Shader linking failed')
        else:
            log.debug('Shader linked.')

        for shaderID in shaderIDs:
            glDeleteShader(shaderID)

    @staticmethod
    def __readFile__(path):
        f = None
        try:
            f = open(path, 'r')
            data = f.read()
            f.close()
            return data
        except IOError as e:
            raise IOError("\"{2}\": I/O error({0}): {1}".format(e.errno, e.strerror, path))
        except:
            raise RuntimeError("Unexpected error: ", sys.exc_info()[0])

    @staticmethod
    def __createShader__(shaderPath, shaderType, shaderCode):
        shader = glCreateShader(shaderType)
        glShaderSource(shader, shaderCode)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log.error(glGetShaderInfoLog(shader))
            raise RuntimeError('[%s]: Shader compilation failed!' % shaderPath)
        else:
            log.debug('Shader compiled (%s).', shaderPath)
        return shader

    def print_info(self):
        print(glGetProgramInterfaceiv(self.__program, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES))
        for i in range(glGetProgramInterfaceiv(self.__program, GL_PROGRAM_OUTPUT, GL_ACTIVE_RESOURCES)):
            name = glGetProgramResourceName(self.__program, GL_PROGRAM_OUTPUT, i, 0)
            params =  glGetProgramResourceiv(self.__program, GL_PROGRAM_OUTPUT, i, 2, [GL_TYPE, GL_LOCATION], 2, 0)
            print('Index %d: %s %s @ location %s' % (i, params[0], name, params[1]))

    def delete(self):
        glDeleteProgram(self.__program)

    def compile_and_use(self):
        self.compile()
        self.use()

    def use(self):
        glUseProgram(self.__program)
        Shader.active_shader = self

    @property
    def id(self):
        return self.__program

    def get_program(self):
        return self.__program