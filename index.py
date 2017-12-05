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

def distance(point1, point2):
    return numpy.linalg.norm(point1 - point2)

def contourLen(contour):
    pass #for (i in )

def main(): 
    fig = pyplot.figure(figsize=(40, 40))
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0)

    for idx in range(len(cards)):
        card_file_name = cards[idx]
        card = cv2.imread(card_file_name)
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1,1), 1000)
        mean = numpy.mean(card)
        print(mean)
        flag, thresh = cv2.threshold(blur, 1.6*mean, 255, cv2.THRESH_BINARY)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_length = max(contours, lambda c: len(c))
        contours = filter(lambda c: len(c) > .1 * len(max_length), contours)
        print(contourLen(contours[0]))
        cv2.drawContours(card, contours, -1, (0,255,0), 7)
        picture = fig.add_subplot(5, 2, idx + 1)
        picture.axis('off')
        pyplot.imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        

    fig.savefig("result.pdf")


main()