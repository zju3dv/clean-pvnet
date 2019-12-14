# -*- coding: UTF-8 -*-
import cyglfw3 as glfw
from OpenGL.GL import *
from OpenGL.GL.NV.bindless_texture import *

class Window(object):

    def __init__(self, window_width, window_height, samples=1, window_title='', monitor=1, show_at_center=True, offscreen=False):
        self.window_title = window_title
        assert glfw.Init(), 'Glfw Init failed!'
        glfw.WindowHint(glfw.SAMPLES, samples)
        if offscreen:
            glfw.WindowHint(glfw.VISIBLE, False);
        mon = glfw.GetMonitors()[monitor] if monitor!=None else None
        self.windowID = glfw.CreateWindow(window_width, window_height, self.window_title, mon)
        assert self.windowID, 'Could not create Window!'
        glfw.MakeContextCurrent(self.windowID)

        if not glInitBindlessTextureNV():
            raise RuntimeError('Bindless Textures not supported')

        self.framebuf_width, self.framebuf_height = glfw.GetFramebufferSize(self.windowID)
        self.framebuffer_size_callback = []
        def framebuffer_size_callback(window, w, h):
            self.framebuf_width, self.framebuf_height = w, h
            for callback in self.framebuffer_size_callback:
                callback(w,h)
        glfw.SetFramebufferSizeCallback(self.windowID, framebuffer_size_callback)
        
        self.key_callback = []
        def key_callback(window, key, scancode, action, mode):
            if action == glfw.PRESS:
                if key == glfw.KEY_ESCAPE:
                    glfw.SetWindowShouldClose(window, True)
            for callback in  self.key_callback:
                callback(key, scancode, action, mode)
        glfw.SetKeyCallback(self.windowID, key_callback)

        self.mouse_callback = []
        def mouse_callback(window, xpos, ypos):
            for callback in self.mouse_callback:               
                callback(xpos, ypos)
        glfw.SetCursorPosCallback(self.windowID, mouse_callback)

        self.mouse_button_callback = []
        def mouse_button_callback(window, button, action, mods):
            for callback in self.mouse_button_callback:
                callback(button, action, mods)
        glfw.SetMouseButtonCallback(self.windowID, mouse_button_callback)

        self.scroll_callback = []
        def scroll_callback( window, xoffset, yoffset ):
            for callback in self.scroll_callback:
                callback(xoffset, yoffset)
        glfw.SetScrollCallback(self.windowID, scroll_callback)

        self.previous_second = glfw.GetTime()
        self.frame_count = 0.0

        if show_at_center:
            monitors = glfw.GetMonitors()
            assert monitor >= 0 and monitor < len(monitors), 'Invalid monitor selected.'
            vidMode = glfw.GetVideoMode(monitors[monitor])
            glfw.SetWindowPos(self.windowID, 
                            vidMode.width/2-self.framebuf_width/2, 
                            vidMode.height/2-self.framebuf_height/2)

    def update_fps_counter(self):
        current_second = glfw.GetTime()
        elapsed_seconds = current_second - self.previous_second
        if elapsed_seconds > 1.0:
            self.previous_second = current_second
            fps = float(self.frame_count) / float(elapsed_seconds)
            glfw.SetWindowTitle(self.windowID, '%s @ FPS: %.2f' % (self.window_title, fps))
            self.frame_count = 0.0
        self.frame_count += 1.0

    def is_open(self):
        return not glfw.WindowShouldClose(self.windowID)

    def swap_buffers(self):
        glfw.SwapBuffers(self.windowID)

    def poll_events(self):
        glfw.PollEvents()

    def update(self):
        self.swap_buffers()
        self.poll_events()
        self.update_fps_counter()    

    def close(self):
        glfw.Terminate()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()