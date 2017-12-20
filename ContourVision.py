import cv2
import numpy as np

def main():
    startCamera()


class ImageAnalysis():

    #FOV info, used for distance calculations
    FOV_HORZ_ANGLE = 60
    FOV_VERT_ANGLE = 60
    FOV_WIDTH_PIXELS = 640
    FOV_HEIGHT_PIXELS = 480
    FOCAL_LENGTH = 700

    #Color filters for the tape (green LED or white light.) BGR format.
    #LOWER_COLOR_BOUNDS = np.array([200, 230, 40], dtype="uint8")  # RGB BGR
    LOWER_COLOR_BOUNDS = np.array([220, 220, 220], dtype="uint8")
    UPPER_COLOR_BOUNDS = np.array([255, 255, 255], dtype="uint8")

    # Coefficient used in side approximation
    GENEROSITY_CONSTANT = 0.02

    # In inches. Go America!
    PEG_TAPE_HEIGHT = 5
    PEG_TAPE_WIDTH = 2
    HIGH_GOAL_TAPES_WIDTH = 15
    TOP_TAPE_HEIGHT = 4
    BOT_TAPE_HEIGHT = 2

    PEG_TAPE_DISTANCE_CENTER2CENTER = 8.25
    PEG_TAPE_DISTANCE_IN_BW = 6.25
    HIGH_GOAL_TAPE_DISTANCE_BW_EDGES = 10

    # Camera feed with drawn contours.
    EDITED_IMAGE = np.zeros((FOV_HEIGHT_PIXELS, FOV_WIDTH_PIXELS, 3), np.uint8)

    #def setImageFrame(self, sourceImage):
    #    self.sourceImage = sourceImage


    def startAnalysis(self, sourceImage):
        # The base for calculation contours and drawing them onto the image

        # Apply a color filter.
        imageMask = cv2.inRange(sourceImage, ImageAnalysis.LOWER_COLOR_BOUNDS, ImageAnalysis.UPPER_COLOR_BOUNDS)
        imageMask = abs(255 - imageMask)
        retVal, threshedImage = cv2.threshold(imageMask, 255, 255, 255)

        # Find all valid contours
        filteredStream, validContours, hierarchy = cv2.findContours(threshedImage.copy(), cv2.RETR_CCOMP,
                                                                    cv2.CHAIN_APPROX_TC89_L1)

        # Center X, Center Y, Area
        previousValidTape = [0, 0, 0]

        # Go through every filter
        for contour in validContours:
            try:
                # Calculations for center.
                moments = cv2.moments(contour)
                centerPoint = (int((moments["m10"] / moments["m00"])),
                               int((moments["m01"] / moments["m00"])))

                # Get perimeter
                contour = contour.astype("int")
                cArea = cv2.contourArea(contour)
                cPerimeter = cv2.arcLength(contour, True)

                # Bounding rectangle
                cBoundary = cv2.approxPolyDP(contour, ImageAnalysis.GENEROSITY_CONSTANT * cPerimeter, True)
                x, y, w, h = cv2.boundingRect(cBoundary)

                # Custom-made method for finding the "shape" of the contour.
                shape = self.determineShape(cBoundary, cPerimeter, w, h)

                # The tape is a rectangle.
                if (shape == "Rectangle"):
                    # Draw on image, write to txt file.
                    self.drawContourMarkers(sourceImage, x, y, w, h, centerPoint, cBoundary)
                    outputFile = open("output.txt", "w")
                    # Determine whether the high or low goal is in view
                    target = self.determineTarget(centerPoint[0], centerPoint[1], cArea,
                                                           previousValidTape[0], previousValidTape[1],
                                                           previousValidTape[2])

                    if (target == "Gear Peg"):
                        # Calculations to get distance from gear peg.
                        midPoint = self.calculateMidpoint(sourceImage, centerPoint[0],
                                                                   previousValidTape[0],
                                                                   centerPoint[1], target)

                        distanceToTarget = (ImageAnalysis.FOCAL_LENGTH * ImageAnalysis.PEG_TAPE_DISTANCE_CENTER2CENTER /
                                            abs(centerPoint[0] - previousValidTape[0]))

                        theta = (midPoint[0] * (ImageAnalysis.FOV_HORZ_ANGLE / 2) /
                                 (ImageAnalysis.FOV_WIDTH_PIXELS / 2) -
                                 (ImageAnalysis.FOV_HORZ_ANGLE / 2))

                        cv2.putText(sourceImage, "Distance: %f" % (distanceToTarget), (x, y - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(sourceImage, "Theta: " + str(theta), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2)

                        outputFile.write("Distance from target:\n%f\n" % (distanceToTarget))
                        outputFile.write("Angle from target:\n%f\n" % (theta))
                        outputFile.close()

                    elif (target == "High Goal"):
                        # Calculations to get distance from top of boiler
                        midPoint = self.calculateMidpoint(sourceImage, centerPoint[1],
                                                                   previousValidTape[1],
                                                                   centerPoint[0], target)

                        distanceToTarget = (ImageAnalysis.FOCAL_LENGTH * ImageAnalysis.HIGH_GOAL_TAPES_WIDTH /
                                            float(w))

                        theta = (midPoint[0] * (ImageAnalysis.FOV_HORZ_ANGLE / 2) /
                                 (ImageAnalysis.FOV_WIDTH_PIXELS / 2) -
                                 (ImageAnalysis.FOV_HORZ_ANGLE / 2))

                        cv2.putText(sourceImage, "Distance: %f" % (distanceToTarget), (x, y - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(sourceImage, "Theta: " + str(theta), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2)

                        outputFile.write("Distance from target:\n%f\n" % (distanceToTarget))
                        outputFile.write("Angle from target:\n%f\n" % (theta))
                        outputFile.close()
                    else:
                        previousValidTape[0] = centerPoint[0]
                        previousValidTape[1] = centerPoint[1]
                        previousValidTape[2] = cArea

            except ZeroDivisionError:
                pass
        return sourceImage

    
    def determineShape(self, boundary, perimeter, w, h):
        shape = "unknown"

        # Too small to be determined
        if (perimeter < 60 or perimeter >= 2000):
            return shape

        # len(approx) returns the number of vertices on the contour's approximated polygon.
        # Compares this value with a preset value to determine what kind of polygon it is
        elif (len(boundary) == 4):
            whRatio = float(w) / h
            shape = "Rectangle"

            if (0.9 <= whRatio <= 1.05):
                shape = "Square"

        return shape

    def drawContourMarkers(self, source, x, y, w, h, center, boundary):
        cv2.circle(source, center, 3, (0, 0, 0), -1)
        cv2.drawContours(source, [boundary], -1, (0, 255, 0), 3)
        cv2.rectangle(source, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # TODO Debug print statements here
        cv2.putText(source, "Area: " + str(w * h), (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(source, "Y Position: " + str(y), (x, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def determineTarget(self, x, y, area, prevX, prevY, prevArea):
        target = "n/a"


        if (0.75 <= float(prevArea) / area <= 1.1 or 0.45 <= float(prevArea) / area <= 0.55):
            if (0.9 <= float(prevY) / y <= 1.1):
                target = "Gear Peg"
            elif (0.9 <= float(prevX) / x <= 1.1):
                target = "High Goal"
        return target

    def calculateMidpoint(self, source, px1, px2, px3, target):

        if (target == "Gear Peg"):
            cv2.circle(source, (px1 + (px2 - px1) / 2, px3), 10, (255, 0, 0), -1)
            return (px1 + (px2 - px1) / 2, px3)
        elif (target == "High Goal"):
            cv2.circle(source, (px3, (px1 + (px2 - px1) / 2)), 10, (255, 0, 0), -1)
            return (px3, px1 + (px2 - px1) / 2)

    def getContouredImage(self):
        return ImageAnalysis.EDITED_IMAGE


def startCamera():
    global VIDEO_STREAM

    for i in range(0, 5):
        VIDEO_STREAM = cv2.VideoCapture(i)
        if (VIDEO_STREAM.isOpened()):
            print("Camera found on port: %d" % (i))
            print("Width resolution: %d" % (VIDEO_STREAM.get(3)))
            print("Height resolution: %d" % (VIDEO_STREAM.get(4)))
            break
        if (not VIDEO_STREAM.isOpened()):
            print("Camera not found!")
            continue
    imageAnalysis = ImageAnalysis()
    while (VIDEO_STREAM.isOpened()):
        ret, source = VIDEO_STREAM.read()
        cv2.imshow("Image", imageAnalysis.startAnalysis(source))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    VIDEO_STREAM.release()
    cv2.destroyAllWindows()

main()
