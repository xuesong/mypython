import cv2
import matplotlib.pyplot as plt


#img = cv2.imread("input.jpg",cv2.IMREAD_GRAYSCALE)

img = cv2.imread("input.jpg")






cv2.imshow("img",img)

#grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("grayimgbgr",grayimg)

#grayimg2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#cv2.imshow("grayimgrgb",grayimg2)

#rgbimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#cv2.imshow("imgrgb",rgbimg)

edges = cv2.Canny(img,50,200)
cv2.imshow("Canny",edges)

cv2.waitKey()
cv2.destroyAllWindows()




# https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0