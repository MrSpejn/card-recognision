import cv2
import numpy

def process_card(card):
    lower_range = numpy.array([15, 15, 100], dtype=numpy.uint8)
    upper_range = numpy.array([60, 60, 255], dtype=numpy.uint8)
    mask = cv2.inRange(card, lower_range, upper_range)
    mask3=sum(sum(mask))
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    mean = numpy.mean(card)
    flag, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(map(lambda c: cv2.contourArea(c), contours))
    contours=filter(lambda c: cv2.contourArea(c) > .6 * max_area, contours)
    avg_approx=0
    count_approx=0
    for c in contours:
        approx = cv2.approxPolyDP(c, .025 * cv2.arcLength(c, True), True)
        avg_approx+=len(approx)
        count_approx+=1
    avg_approx/=count_approx
    mes=""
    if(avg_approx<4.5):
        mes=" karo"
    elif(mask3>0):
        mes=" kier"
    elif(avg_approx<10):
        mes=" pik"
    else:
        mes=" trefl"
    cv2.drawContours(card, contours, -1, (0,255,0), 12)
    cv2.putText(card,str(count_approx)+mes,(25,card.shape[0]-33), cv2.FONT_HERSHEY_SIMPLEX, 5,(0,0,0),10)
    return card
