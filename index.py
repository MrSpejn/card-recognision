from matplotlib import pyplot
import cv2
import numpy
import math
from card_processing import process_card

def order_points(points):
    rect = numpy.zeros((4, 2), dtype = "float32")
    rect2 = numpy.zeros((4, 2), dtype = "float32")
    s = points.sum(axis = 2)
    rect[2] = points[numpy.argmax(s)]
    rect[0] = points[numpy.argmin(s)]
    diff = numpy.diff(points, axis = 2)
    rect[1] = points[numpy.argmin(diff)]
    rect[3] = points[numpy.argmax(diff)]

    x,y,w,h = cv2.boundingRect(rect)

    if (h > 0.9*w): return rect
    else:
        rect2[0] = rect[3]
        rect2[1] = rect[0]
        rect2[2] = rect[1]
        rect2[3] = rect[2]
        return rect2

def extract_poly(contour):
    epsilon = 0.05*cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def extract_rect(contour):
    rect = cv2.minAreaRect(contour)
    return numpy.int0(map(lambda i: [i], cv2.boxPoints(rect)))

def filterout_bg(image, foreground, add=False):
    mask = numpy.zeros_like(image)
    out = numpy.zeros_like(image)
    kernel = numpy.ones((50,50), numpy.uint8)
    cv2.drawContours(mask, [foreground], -1, [255, 255, 255], -1)
    if (add):
        mask = cv2.dilate(mask, kernel, iterations = 4)
    out[mask == 255] = image[mask == 255] 
    return out
    
def transform_rect(image, rect, add=False):
    rect = order_points(rect)
    rect = rect.astype(numpy.float32)

    width = 750
    height = 1000
    h = numpy.array([[0,0],[749,0],[749,999],[0,999]], numpy.float32)

    if (add):
        h = numpy.array([[25,25],[774,25],[774,1024],[25,1024]], numpy.float32)
        width = 800
        height = 1050

    transform = cv2.getPerspectiveTransform(rect, h)
    return cv2.warpPerspective(image, transform, (width, height))

def get_contoursHSV(image, factor, area, use_mean=True):
    height, width, _ = image.shape
    size = max(width, height)
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mean1 = numpy.mean(s) if use_mean else 1
    mean2 = numpy.mean(v) if use_mean else 1

    out = numpy.zeros_like(image)
    out2 = numpy.zeros_like(image)

    flag, thresh1 = cv2.threshold(s, 110, 255, cv2.THRESH_BINARY_INV)
    flag, thresh2 = cv2.threshold(v, mean2, 255, cv2.THRESH_BINARY)

    thresh = cv2.bitwise_and(thresh1, thresh2)

    kernel = numpy.ones((10,10), numpy.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(map(lambda c: cv2.contourArea(c), contours))
    contours = filter(lambda c: cv2.contourArea(c) > area * max_area, contours)
    return contours
    
def get_contours(image, factor, area, use_mean=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 1000)
    mean = numpy.mean(gray) if use_mean else 1
    flag, thresh = cv2.threshold(blur, factor*mean, 255, cv2.THRESH_BINARY)    

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(map(lambda c: cv2.contourArea(c), contours))
    contours = filter(lambda c: cv2.contourArea(c) > area * max_area, contours)
    return contours

def extract_card(image, contour):
    boudingQuad = extract_poly(contour)
    foreground = transform_rect(filterout_bg(image, boudingQuad), boudingQuad)

    return foreground

def main(): 
    fig = pyplot.figure(figsize=(40, 40))
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0)

    for idx in range(10, 20):
        card = cv2.imread("./cardsv2/card"+str(idx)+".JPG")
        contours = get_contoursHSV(card, 1.45, .3, True)    
        picture = fig.add_subplot(20, 5, 5*(idx-10) + 1)
        picture.axis('off')
        pyplot.imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        for ci in range(len(contours)):
            single_card_rect = extract_card(card, contours[ci])
            if (len(single_card_rect) == 0): continue
            processed = process_card(single_card_rect)            
            picture = fig.add_subplot(20, 5, 5*(idx-10) + 2 + ci)
            picture.axis('off')
            pyplot.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    fig.savefig("result.pdf")

main()
