import cv2
import numpy
import math

def get_color(card, shape):
    shape_x, shape_y, w, h = cv2.boundingRect(shape)
    return card[shape_y + int(h/2), shape_x + int(w/2)]



def process_card(card):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    mean = numpy.mean(card)

    width, height, channels = card.shape

    flag, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(map(lambda c: cv2.contourArea(c), contours))
    contours=filter(lambda c: cv2.contourArea(c) > .6 * max_area, contours)
    avg_approx=0

    
    for c in contours:
        approx = cv2.approxPolyDP(c, .015 * cv2.arcLength(c, True), True)
        avg_approx+=len(approx)

    if (any(cv2.contourArea(c) > 0.1 * width * height for c in contours)):
        cv2.putText(card, "Unable to detect", (25,card.shape[0]-33), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,0,0), 10)
        return card

    redness = numpy.mean(map(lambda c: get_color(card, c)[2], contours))
    print redness

    avg_approx /= len(contours)
    mes=""
    if (redness > 200):
        if (avg_approx < 9):
            mes=" karo"
        else :
            mes=" kier"
    else:
        if(avg_approx < 12):
            mes=" pik"
        else:
            mes=" trefl"
        
    cv2.drawContours(card, contours, -1, (0,255,0), 12)
    cv2.putText(card, str(len(contours)) + mes, (25,card.shape[0]-33), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 10)
    return card
