import os

cuda_include='/usr/local/cuda-9.0/include'
os.system('nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.o -x cu -Xcompiler -fPIC -O2 -arch=sm_52 -I {}'.format(cuda_include))

from cffi import FFI
ffibuilder = FFI()


with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/nearest_neighborhood.cu.o',
                   '/usr/local/cuda-9.0/lib64/libcudart.so'],
    libraries=['stdc++']
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
    os.system("rm *.o")
