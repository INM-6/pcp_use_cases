import cython_lib as cython_lib

import numpy as np

array = np.ones((2,2,2),dtype=np.float32)

array[0,0,0] = 20
array[1,1,1] = 100


return_value = cython_lib.accelerated.log_approx_array(array)

print return_value