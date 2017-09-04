import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import scipy.misc
from skimage import data

lena = data.clock()
rotated_lena = rotate(lena, 30, reshape=True, mode='nearest')

f, (ax0,ax1) = plt.subplots(1,2)
ax0.imshow(lena, cmap='gray')
ax1.imshow(rotated_lena, cmap='gray')
plt.show()