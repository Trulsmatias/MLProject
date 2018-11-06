"""import PIL.Image
import io
import IPython.display
import numpy as np
def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.(data=f.getvalue()))
"""

import util
model = util.load_from_file("../best_model.h5")
print("her")