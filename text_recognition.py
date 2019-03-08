# import the necessary packages
import csv
import os

from imutils.object_detection import non_max_suppression
from methods import get_cart_number, decode_predictions, get_date
import numpy as np
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image folder")
ap.add_argument("-east", "--east", type=str, default='frozen_east_text_detection.pb',
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
# ap.add_argument("-w", "--width", type=int, default=320,
#                 help="nearest multiple of 32 for resized width")
# ap.add_argument("-e", "--height", type=int, default=320,
#                 help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.1,
                help="amount of padding to add to each border of ROI")
ap.add_argument("-iterh", "--iterh", type=int, default=30,
                help="h iteration")
ap.add_argument("-croprate", "--croprate", type=int, default=0.04,
                help="crop rate")
args = vars(ap.parse_args())

from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(args['image']) if isfile(join(args['image'], f))]

with open('redhat_results.csv', mode='w') as redhat_result:
    redhat_writer = csv.writer(redhat_result, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for my_path in onlyfiles:
        if not os.path.exists(args['image'] + my_path):
            print('Image does not exists. Exiting ...')
            continue

        image = cv2.imread(args['image'] + my_path)
        # kernel = np.ones((2, 2), np.uint8)
        # image = cv2.erode(image, kernel)

        m_height, m_width, _ = image.shape
        height_temp_par = m_height - (m_height % 32)
        width_temp_par = m_width - (m_width % 32)
        height_par = min(256, height_temp_par)
        width_par = min(416, width_temp_par)
        crop_height = float(args['croprate']) * m_height
        crop_width = float(args['croprate']) * m_width

        image = image[int(crop_height): int(m_height - crop_height), int(crop_width): int(m_width - crop_width)]
        m_height, _, _ = image.shape
        orig = image.copy()
        (origH, origW) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (width_par, height_par)
        rW = origW / float(newW)
        rH = origH / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))

        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        net = cv2.dnn.readNet(args["east"])

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        # blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), swapRB=False, crop=False)

        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry, args['min_confidence'])
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []

        # copy of modified image
        modified_image = image.copy()

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective ratios
            # startX = int(startX * rW)
            # startY = int(startY * rH)
            # endX = int(endX * rW)
            # endY = int(endY * rH)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            # endX = min(newW, endX + (dX * 2))
            # endY = min(newH, endY + (dY * 2))
            endX = min(newW, endX + dX)
            endY = min(newH, endY + dY)

            # extract the actual padded ROI
            roi = image[startY:endY, startX:endX]
            # cv2.rectangle(modified_image, (startX, startY), (endX, endY), (255, 150, 100), 1)
            # cv2.imshow('Text Detection', cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            # cv2.waitKey(0)

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = "-l eng --oem 2 --psm 7"
            # cv2.imshow('dd', erosion)
            # cv2.waitKey(0)
            (thresh, im_bw) = cv2.threshold(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 128, 255,
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(im_bw, config=config)
            # cv2.imshow('dd', im_bw)
            # cv2.waitKey(0)
            # text = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), config=config)
            # print(text)
            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))

        # sort the results bounding box coordinates from top to bottom
        # results = sorted(results, key=lambda r: r[0][1])

        final_results = []
        passed = []
        my_size = 0
        for y in range(0, m_height, args['iterh']):
            temp = []
            for r in results:
                if y <= r[0][1] < y + args['iterh'] and r not in passed:
                    temp.append(r)
                    passed.append(r)
            if len(temp) > 0:
                my_size += len(temp)
                final_results.append(temp)

        # loop over the results
        for element in final_results:
            element = sorted(element, key=lambda r: r[0][0])
            for ((startX, startY, endX, endY), text) in element:
                # strip out non-ASCII text so we can draw the text on the image
                # using OpenCV, then draw the text and a bounding box surrounding
                # the text region of the input image
                text = "".join([c if ord(c) < 128 else "?" for c in text]).strip()
                cv2.rectangle(modified_image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(modified_image, text, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                # show the output image
                # cv2.imshow("Text Detection", modified_image)

        cart_number_detected = False
        date_detected = False

        number = []
        month = []
        year = []

        for elements in final_results:
            elements = sorted(elements, key=lambda r: r[0][0])
            if not cart_number_detected:
                temp = get_cart_number(elements)
                if temp is not None:
                    number = temp
            elif not date_detected:
                year, month = get_date(elements)
                if month is not None or year is not None:
                    date_detected = True

        redhat_writer.writerow([my_path, ''.join(map(str, number)), ''.join(map(str, year)), ''.join(map(str, month))])
        print('input:', my_path)
        if len(number) > 0:
            print('cart number', end=':')
        for x in number:
            print(x, end='')
        if len(number) > 0:
            print('')

        if len(year) > 0:
            print('year', end=':')
        for x in year:
            print(x, end='')
        if len(year) > 0:
            print('')

        if len(month) > 0:
            print('month', end=':')
        for x in month:
            print(x, end='')
        if len(month) > 0:
            print('')

        if len(number) == len(month) == len(year) == 0:
            print("Failed to recognize...")

redhat_result.close()
