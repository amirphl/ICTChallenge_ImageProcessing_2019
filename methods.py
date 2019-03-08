import numpy as np


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


# TODO
def get_cart_number(array):
    number = []
    for element in array:
        for c in element[1]:
            if 48 <= ord(c) <= 57:
                number.append(c)
            elif c == 'o':
                number.append(0)
            elif c == 'i' or c == 'l':
                number.append(1)
            elif c == '@':
                number.append(5)
            elif c == 's' or c == '&':
                number.append(8)
            elif c == 'b':
                number.append(6)
            elif c == 'm':
                number.append(1)
                number.append(2)
            elif c == ',':
                number.append(9)
    if len(number) == 16:
        return number
    return None


def get_date(array):
    month = []
    year = []
    year_detected = False
    month_detected = False

    for element in array:
        text = element[1]
        number = []
        for i in range(0, len(text)):
            if 48 <= ord(text[i]) <= 57:
                number.append(text[i])
            elif text[i] == 'o':
                number.append('0')
            elif text[i] == 'i' or text[i] == 'l':
                number.append('1')
            elif text[i] == '@':
                number.append('5')
            elif text[i] == 's' or text[i] == '&':
                number.append('8')
            elif text[i] == 'b':
                number.append('6')
            elif text[i] == 'm':
                number.append('1')
                number.append('2')
            elif text[i] == ',':
                number.append('9')
            else:
                number.append(text[i])
        flag = True
        if len(number) == 7:
            for i in range(0, 7):
                if i != 4 and not (48 <= ord(number[i]) <= 57):
                    flag = False
                    break
        if flag:
            if not year_detected:
                y1 = ''.join(map(str, number[0:4]))
                try:
                    y1 = int(y1)
                    if 1400 <= y1 <= 1405:
                        year_detected = True
                        month_detected = True
                        year = str(y1).split()
                except ValueError:
                    pass
            if not month_detected:
                m1 = ''.join(map(str, number[5:len(number)]))
                try:
                    m1_copy = m1
                    m1 = int(m1)
                    if 1 <= m1 <= 12:
                        year_detected = True
                        month_detected = True
                        month = m1_copy.split()
                except ValueError:
                    pass

        if year_detected or month_detected:
            return year, month

        flag = True
        if len(number) == 5:
            for i in range(0, 5):
                if i != 2 and not (48 <= ord(number[i]) <= 57):
                    flag = False
                    break
        if flag:
            if not year_detected:
                y1 = ''.join(map(str, number[0:2]))
                try:
                    y1 = int(y1)
                    if 80 <= y1 <= 99:
                        year_detected = True
                        month_detected = True
                        year = str(y1).split()
                except ValueError:
                    pass
            if not month_detected:
                m1 = ''.join(map(str, number[3:len(number)]))
                try:
                    m1_copy = m1
                    m1 = int(m1)
                    if 1 <= m1 <= 12:
                        year_detected = True
                        month_detected = True
                        month = m1_copy.split()
                except ValueError:
                    pass
        if year_detected or month_detected:
            return year, month
    return year, month
