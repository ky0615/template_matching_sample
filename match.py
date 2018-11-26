import cv2


def main():
    image = cv2.imread("./image/base.png")
    matching_image = cv2.imread("./image/matching.png", -1)

    print(type(matching_image))

    cv2.imshow('base image', image)
    cv2.imshow('matching image', matching_image)

    matchSSD(image, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
             cv2.cvtColor(matching_image, cv2.COLOR_RGB2GRAY))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def matchSSD(base_image, input_image, template_image):
    h, w = template_image.shape

    match = cv2.matchTemplate(input_image, template_image, cv2.TM_SQDIFF)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = min_pt

    cv2.rectangle(base_image, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv2.imshow("SSD result", base_image)


if __name__ == '__main__':
    main()
