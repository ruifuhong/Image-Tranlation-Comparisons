import cv2
import numpy as np 
import os


input_path = './monet2photo/testB/'
output_path = './monet2photo/sobel_test/'
folder = os.listdir(input_path)
for name in folder[1:]:
    img = cv2.imread(input_path+name)
    # data = np.array(img, dtype=float)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    
    x = cv2.Sobel(img, cv2.CV_16S,1,0)
    y = cv2.Sobel(img, cv2.CV_16S,0,1)
    
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
 
    cv2.imwrite(output_path+name, dst)



# cv2.waitKey(0)
# cv2.destroyAllWindows()