from matplotlib import pyplot
import cv2
import numpy

# cards = [
#     './cards/cards1.png',
#     './cards/cards2.jpg',
#     './cards/cards3.jpg',
#     './cards/cards4.jpg',
#     './cards/cards5.jpg',
#     './cards/cards6.jpg',
#     './cards/cards7.jpg',
#     './cards/cards8.jpg',
#     './cards/cards9.jpg',
#     './cards/cards10.jpg',
# ]
cards = [
    './cardsv2/cards1.png',
    './cardsv2/cards2.jpg',
    './cardsv2/card3.jpg',
    './cardsv2/card4.jpg',
    './cardsv2/card5.JPG',
    './cardsv2/card6.JPG',
    './cardsv2/card7.JPG',
    './cardsv2/card8.JPG',
    './cardsv2/card9.JPG',
    './cardsv2/card11.JPG',
    './cardsv2/card12.JPG',
    './cardsv2/card13.JPG',
    './cardsv2/card14.JPG',
    './cardsv2/card15.JPG',
]

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

def extract_rect(image, contours, idx):
    epsilon = 0.01*cv2.arcLength(contours[idx], True)
    approx = cv2.approxPolyDP(contours[idx], epsilon, True)
    approx = order_points(approx)
    approx = approx.astype(numpy.float32)

    h = numpy.array([[0,0],[749,0],[749,999],[0,999]], numpy.float32)
    transform = cv2.getPerspectiveTransform(approx, h)

    approx = approx.astype(numpy.int32)
    mask = numpy.zeros_like(image)
    cv2.drawContours(mask, contours, idx, [255, 255, 255], -1)
    out = numpy.zeros_like(image)
    out[mask == 255] = image[mask == 255]
    wrap = cv2.warpPerspective(out, transform, (750, 1000))
    return wrap


def main(): 
    fig = pyplot.figure(figsize=(40, 40))
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0)

    for idx in range(len(cards)):
        card_file_name = cards[idx]
        card = cv2.imread(card_file_name)
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1,1), 1000)
        mean = numpy.mean(card)
        flag, thresh = cv2.threshold(blur, 1.45*mean, 255, cv2.THRESH_BINARY)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max(map(lambda c: cv2.contourArea(c), contours))
        contours = filter(lambda c: cv2.contourArea(c) > .1 * max_area, contours)
        
      

        picture = fig.add_subplot(16, 5, 5*idx + 1)
        picture.axis('off')
        pyplot.imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        for ci in range(len(contours)):
            single_card_rect = extract_rect(card, contours, ci)
            picture = fig.add_subplot(16, 5, 5*idx + 2 + ci)
            picture.axis('off')
            pyplot.imshow(cv2.cvtColor(single_card_rect, cv2.COLOR_BGR2RGB))

    fig.savefig("result.pdf")



main()