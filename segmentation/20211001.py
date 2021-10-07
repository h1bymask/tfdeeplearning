import cv2
import numpy as np

def cv_show(name, image):
    cv2.namedWindow(name, 0)
    cv2.startWindowThread()
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('input.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
ret, binary = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
back_mask = cv2.erode(binary, kernel, iterations=2)

numbers_mask = cv2.bitwise_not(back_mask)
# Median filter
numbers_mask = cv2.medianBlur(numbers_mask,3)
cv_show("", numbers_mask)

# Find contour
contours, hierarchy = cv2.findContours(numbers_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
def zc_sort(a):
    l = len(a)
    j = 0
    for i in range(1, l):
        (temp, y, w, h) = cv2.boundingRect(a[i])
        temp_ = a[i]
        for j in range(i - 1, -1, -1):
            (aj_temp, y1, w1, h1) = cv2.boundingRect(a[j])
            if temp < aj_temp:   # If the i-th element is greater than the j-th element in the previous i,
                a[j + 1] = a[j]  # then the j-th element is shifted by 1 place
                a[j] = temp_     # Assign i elements to empty positions
            else:      # If the i-th element is less than or equal to the j-th of the previous i elements, the loop ends
                break
    return a
contours = zc_sort(contours)

canvas = cv2.cvtColor(numbers_mask, cv2.COLOR_GRAY2BGR)
def getStandardDigit(img):
    STD_WIDTH = 32  # Standard width
    STD_HEIGHT = 64
    height,width = img.shape
    # Determine whether there is a long 1
    new_width = int(width * STD_HEIGHT / height)
    if new_width > STD_WIDTH:
        new_width = STD_WIDTH
    # Scale based on height
    resized_num = cv2.resize(img, (new_width,STD_HEIGHT), interpolation = cv2.INTER_NEAREST)
    # New canvas 
    canvas = np.zeros((STD_HEIGHT, STD_WIDTH))
    x = int((STD_WIDTH - new_width) / 2)
    canvas[:,x:x+new_width] = resized_num
    return canvas

minWidth = 5    # Minimum width
minHeight = 55  # Minimum height
maxHeight = 300

base = 1000    # Count number
imgIdx = base  # The number of the current picture

# Retrieve the area that meets the conditions
for cidx,cnt in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(cnt)
    if w < minWidth or h < minHeight or h > maxHeight:
        # Filter out if the conditions are not met
        continue
    # Get picture of the ROI
    digit = numbers_mask[y:y+h, x:x+w]
    # print(digit.shape)
    digit = getStandardDigit(digit)
    # cv2.imwrite('count/{}.png'.format(imgIdx), digit)
    # cv_show('', digit)
    imgIdx+=1
    # Paint a bounding box around the original canvas
    cv2.rectangle(canvas, pt1=(x, y), pt2=(x+w, y+h),color=(0, 255, 255), thickness=2)
cv2.imwrite('output.png', canvas)
