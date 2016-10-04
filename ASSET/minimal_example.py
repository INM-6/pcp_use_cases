import numpy as np

import quantities as pq


print ("Using floats:")
np.linspace(0.0, 1.0 , 20 + 1, endpoint=True)
print ("success \n")

start_pq = 0 * pq.s
end_pq = 1 * pq.s

#try:
#    print ("Using quantities:")
np.linspace(start_pq, end_pq,  20 + 1, endpoint=True)
#    print ("success \n")

#except Exception, e:
#    print ("fail with error: \n" + str(e))


