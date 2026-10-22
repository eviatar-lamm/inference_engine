import numpy as np
import time

a = np.random.random((3082, 368))
now = time.time()

a @ a.T

print(f"Took {time.time() - now} s")

