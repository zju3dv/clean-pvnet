import os

cuda_include = os.path.join(os.environ.get('CUDA_HOME'), 'include')
# cmd :
# nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.obj -x cu -Xcompiler "/MD /EHsc /O2" -arch=sm_75 -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\include"
os.system('nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.obj -x cu -Xcompiler "/MD /EHsc /O2" -arch=sm_75 -I "{}"'.format(cuda_include))

from cffi import FFI

ffibuilder = FFI()

with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/nearest_neighborhood.cu.obj',
                   os.path.join(os.environ.get('CUDA_HOME'), 'lib/x64/cudart.lib')],
    # libraries=['stdc++']
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("del /q src\*.obj")
    os.system("del /q *.obj")
