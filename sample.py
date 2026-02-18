
import pycuda.driver as cuda
import pycuda.autoinit 
import numpy

from pycuda.compiler import SourceModule

a = numpy.random.randn(4, 4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doubleit(float* a) {
        int idx = threadIdx.x + threadIdx.y * 4;
        a[idx] *= 2;
    }        
""")

mod2 = SourceModule("""
    __global__ void thripleit(float* a) {
        int idx = threadIdx.x + threadIdx.y * 4;
        a[idx] *= 2;
    }
""")

func = mod.get_function("doubleit")
func2 = mod2.get_function("thripleit")

func(a_gpu, block=(4, 4, 1))

a_doubled = numpy.empty_like(a)

cuda.memcpy_dtoh(a_doubled, a_gpu)

print(a_doubled)

print(a)

func2(a_gpu, block=(4,4,1))

a_thripled = numpy.empty_like(a)

cuda.memcpy_dtoh(a_thripled, a_gpu)

print(a_thripled)