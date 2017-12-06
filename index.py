from matplotlib import pyplot
import cv2
import numpy

cards = [
    './cards/cards1.png',
    './cards/cards2.jpg',
    './cards/cards3.jpg',
    './cards/cards4.jpg',
    './cards/cards5.jpg',
    './cards/cards6.jpg',
    './cards/cards7.jpg',
    './cards/cards8.jpg',
    './cards/cards9.jpg',
    './cards/cards10.jpg',
]
def order_points(points):
    rect = numpy.zeros((4, 2), dtype = "float32")
    s = points.sum(axis = 2)
    rect[2] = points[numpy.argmax(s)]
    rect[0] = points[numpy.argmin(s)]
    diff = numpy.diff(points, axis = 2)
    rect[1] = points[numpy.argmin(diff)]
    rect[3] = points[numpy.argmax(diff)]
    return rect

def main(): 
    fig = pyplot.figure(figsize=(40, 40))
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0)

    for idx in range(len(cards)):
        card_file_name = cards[idx]
        card = cv2.imread(card_file_name)
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1,1), 1000)
        mean = numpy.mean(card)
        flag, thresh = cv2.threshold(blur, 1.6*mean, 255, cv2.THRESH_BINARY)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max(map(lambda c: cv2.contourArea(c), contours))
        contours = filter(lambda c: cv2.contourArea(c) > .1 * max_area, contours)
        
      

        picture = fig.add_subplot(11, 5, 5*idx + 1)
        picture.axis('off')
        pyplot.imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        for ci in range(len(contours)):
            epsilon = 0.01*cv2.arcLength(contours[ci], True)
            approx = cv2.approxPolyDP(contours[ci], epsilon, True)
            approx = order_points(approx)
            approx = approx.astype(numpy.float32)

            h = numpy.array([[0,0],[999,0],[999,999],[0,999]], numpy.float32)
            transform = cv2.getPerspectiveTransform(approx, h)

            approx = approx.astype(numpy.int32)
            mask = numpy.zeros_like(card)
            cv2.drawContours(mask, contours, ci, [255, 255, 255], -1)
            out = numpy.zeros_like(card)
            out[mask == 255] = card[mask == 255]
            wrap = cv2.warpPerspective(out, transform, (1000, 1000))

            picture = fig.add_subplot(11, 5, 5*idx + 2 + ci)
            picture.axis('off')
            pyplot.imshow(cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB))

    fig.savefig("result.pdf")



main()