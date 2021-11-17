import os

ceres_include=r'C:\Program Files\Ceres\include'
ceres_library=r'C:\Program Files\Ceres\lib\ceres.lib'
eigen_include=r'./include/eigen3'
glog_include=r'C:\Program Files\glog\include'
glog_library=r'C:\Program Files\glog\lib\glog.lib'
gflags_include=r'C:\Program Files\gflags\include'
gflags_library=r'C:\Program Files\gflags\lib\gflags.lib'
cxsparse_library=r'D:\RainYQ\CXSparse\build\Release\cxsparse.lib'
suitesparse_library=[r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\libspqr.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\libcholmod.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\libccolamd.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\libcamd.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\libcolamd.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\libamd.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\suitesparseconfig.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\build\install\lib\metis.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\lapack_windows\x64\liblapack.lib',
                     r'D:\RainYQ\suitesparse-metis-for-windows\lapack_windows\x64\libblas.lib']

# os.system('gcc -shared src/uncertainty_pnp.cpp -c -o src/uncertainty_pnp.cpp.o -fopenmp -fPIC -O2 -std=c++11 -I {} -I {}'.format(ceres_include,eigen_include))
os.system('cl.exe src/uncertainty_pnp.cpp /c /Fo"src/uncertainty_pnp.cpp.obj" /MD /EHsc /openmp /O2 /std:c++14 -I "{}" -I "{}" -I "{}" -I "{}"'.format(gflags_include, glog_include,ceres_include,eigen_include))

from cffi import FFI
ffibuilder = FFI()


with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source(
    "_ext",
    """
    #include "src/ext.h"
    """,
    extra_objects=['src/uncertainty_pnp.cpp.obj',
                   ceres_library,
                   glog_library,
                   cxsparse_library] + suitesparse_library
    # libraries=['stdc++']
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("del /q src\*.obj")
    os.system("del /q *.obj")
