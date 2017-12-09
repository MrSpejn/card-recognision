import cv2
import numpy

def getContours(card):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    mean = numpy.mean(card)
    flag, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(map(lambda c: cv2.contourArea(c), contours))
    return filter(lambda c: cv2.contourArea(c) > .6 * max_area, contours)

def process_card(card):
    contours = getContours(card)
    cv2.drawContours(card, contours, -1, (0,255,0), 12)
    return card