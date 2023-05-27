
import cv2
from conf import *


print('Creation layout pattern')

base = cv2.imread('base.png')

thickness = 5

res = cv2.rectangle(base, (MARGIN - thickness, MARGIN - thickness),
                          (base.shape[1] - MARGIN + thickness, base.shape[0] - MARGIN + thickness),
                          (0, 0, 0), thickness)

cv2.imwrite('layout.png', res)
