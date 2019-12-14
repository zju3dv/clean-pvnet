# -*- coding: utf-8 -*-
import logging as log
import os

from OpenGL.GL import *

import cyglfw3 as glfw

class OffscreenContext(object):

    def __init__(self):
        assert glfw.Init(), 'Glfw Init failed!'
        glfw.WindowHint(glfw.VISIBLE, False);
        self._offscreen_context = glfw.CreateWindow(1, 1, "", None)
        assert self._offscreen_context, 'Could not create Offscreen Context!'
        glfw.MakeContextCurrent(self._offscreen_context)

        self.previous_second = glfw.GetTime()
        self.frame_count = 0.0
        self._fps = 0.0

    def update(self):
        self.poll_events()
        self.update_fps_counter()

    def poll_events(self):
        glfw.PollEvents()

    def update_fps_counter(self):
        current_second = glfw.GetTime()
        elapsed_seconds = current_second - self.previous_second
        if elapsed_seconds > 1.0:
            self.previous_second = current_second
            self._fps = float(self.frame_count) / float(elapsed_seconds)
            self.frame_count = 0.0
        self.frame_count += 1.0

    @property
    def fps(self):
        return self._fps

    def close(self):
        glfw.Terminate()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

