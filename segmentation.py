import cv2
import sys
import numpy
from PIL import Image

def cv_show(name, image):
    cv2.namedWindow(name, 0)
    cv2.startWindowThread()
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sort_LTR(images):
    l = len(images)
    j = 0
    for i in range(1, l):
        (temp, y, w, h) = cv2.boundingRect(images[i])
        temp_ = images[i]
        for j in range(i - 1, -1, -1):
            (aj_temp, y1, w1, h1) = cv2.boundingRect(images[j])
            if temp < aj_temp:   # If the i-th element is greater than the j-th element in the previous i,
                images[j + 1] = images[j]  # then the j-th element is shifted by 1 place
                images[j] = temp_     # Assign i elements to empty positions
            else:      # If the i-th element is less than or equal to the j-th of the previous i elements, the loop ends
                break
    return images


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
    canvas = numpy.zeros((STD_HEIGHT, STD_WIDTH))
    x = int((STD_WIDTH - new_width) / 2)
    canvas[:,x:x+new_width] = resized_num
    return canvas

def digit_segmentation(img):
    if (isinstance(img, Image.Image)):
        img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    ret, binary = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    back_mask = cv2.erode(binary, kernel, iterations=2)

    numbers_mask = cv2.bitwise_not(back_mask)
    # Median filter
    numbers_mask = cv2.medianBlur(numbers_mask,3)
    #cv_show("mask", numbers_mask)

    # Find contour
    contours, hierarchy = cv2.findContours(numbers_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_LTR(contours)

    canvas = cv2.cvtColor(numbers_mask, cv2.COLOR_GRAY2BGR)

    minWidth = 5    # Minimum width
    minHeight = 55  # Minimum height
    maxHeight = 300
    padding = 10


    # Retrieve the area that meets the conditions
    for contourindex, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if w < minWidth or h < minHeight or h > maxHeight:
            # Filter out if the conditions are not met
            continue
        # Get a picture of the ROI
        digit = numbers_mask[y:y+h, x:x+w]
        # print(digit.shape)
        digit = getStandardDigit(digit)
        digit = cv2.copyMakeBorder(digit, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        # Convert to MNIST format
        digit = cv2.resize(digit, (28, 28))
        #digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)  # Already a BW image
        yield digit
        # Debug: paint a bounding box around the digit on the original canvas
        cv2.rectangle(canvas, pt1=(x, y), pt2=(x+w, y+h),color=(0, 255, 255), thickness=2)
    cv2.imwrite('segmented.png', canvas)

# If called directly
if 'segmentation' not in sys.modules:
    print('modules:', sys.modules)
    # Write results
    base = 1000    # Count number
    imgIdx = base  # The number of the current picture
    for index, digit in enumerate(digit_segmentation(cv2.imread('mnist/input.png'))):
        cv2.imwrite('segments/{}.png'.format(imgIdx), digit)
        #cv_show('digit', digit)
        imgIdx+=1
