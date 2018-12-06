import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    # image = cv2.imread("./image/base.png")
    # image = cv2.imread("/Users/tarosa/workspace/education/cvTest/P1180635.jpg")
    # matching_image = cv2.imread("./image/matching.png")

    image = cv2.imread("./image/nak/image3.png")
    matching_image = cv2.imread("./image/nak/template4.png")

    print(type(matching_image))

    # cv2.imshow('base image', image)
    # cv2.imshow('matching image', matching_image)

    input_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template_image = cv2.cvtColor(matching_image, cv2.COLOR_RGB2GRAY)

    # template_matching_ssd_builtin(image.copy(), input_image, template_image)
    template_matching_builtin(image, input_image, template_image)
    # template_matching_ssd_original(image.copy(), input_image, template_image)
    # tplot()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def template_matching_builtin(base_image, input_image, template_image):
    fileName = [
        'CCOEFF',
        'CCOEFF_NORMED',
        'CCORR',
        'CCORR_NORMED',
        'SQDIFF',
        'SQDIFF_NORMED'
    ]
    methods = [
        'cv2.TM_CCOEFF',
        'cv2.TM_CCOEFF_NORMED',
        'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED',
        'cv2.TM_SQDIFF',
        'cv2.TM_SQDIFF_NORMED'
    ]

    thresholds = [
        0.9,
        0.9,
        0.9,
        0.97,
        0.05,
        0.05
    ]

    # methods = [methods[3]]
    # thresholds = [0.97]

    h, w = template_image.shape

    score = np.empty((len(methods), 2))

    for i in range(0, len(methods)):
        start = time.time()

        img = input_image.copy()
        b_img = base_image.copy()
        method = eval(methods[i])
        match = cv2.matchTemplate(img, template_image, method)
        threshold = thresholds[i]

        min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
        score[i][0] = min_value
        score[i][1] = max_value

        elapsed_time = time.time() - start
        print("{}, elapsed_time:{}".format(methods[i], elapsed_time) + "[sec]")

        # print(methods[i])
        # print("min: {}, max: {}".format(min_value, max_value))
        # print(match)
        # print((match - min_value) / max_value)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:

            loc = np.where((match - min_value) / max_value <= threshold)
            m_pt = min_pt
        else:
            loc = np.where(match / max_value >= threshold)
            m_pt = max_pt

        c = 5000
        for pt in zip(loc[1][0:c][::-1], loc[0][0:c][::-1]):
            cv2.rectangle(b_img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)

        cv2.rectangle(b_img, (m_pt[0], m_pt[1]), (m_pt[0] + w, m_pt[1] + h), (200, 0, 0), 3)

        plt.figure(figsize=(19, 6))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.suptitle(methods[i])
        plt.tight_layout()
        plt.subplot(1, 2, 1), plt.imshow(match, cmap=plt.cm.Blues)

        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()
        # plt.savefig('./figure/{}.png'.format(fileName[i]))
    print(score)


def template_matching_ssd_builtin(base_image, input_image, template_image):
    h, w = template_image.shape

    match = cv2.matchTemplate(input_image, template_image, cv2.TM_CCOEFF_NORMED)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = min_pt

    cv2.rectangle(base_image, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv2.imshow("SSD result", base_image)


def template_matching_ssd_original(base_image, input_image, template_image):
    # 画像の高さ
    h, w = input_image.shape
    ht, wt = template_image.shape

    # スコア保存用配列
    score = np.empty((h - ht, w - wt))

    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            # 誤差の二乗和を計算
            diff = (input_image[dy:dy + ht, dx:dx + wt] - template_image) ** 2
            score[dy, dx] = diff.sum()

    fig, ax = plt.subplots()
    surf = ax.pcolor(np.fliplr(np.roll(np.rot90(np.rot90(score)), 1, axis=0)), cmap=plt.cm.Blues)
    fig.colorbar(surf)
    ax.set_title("SSD")
    fig.show()

    print(score.argmin())
    pt = np.unravel_index(score.argmin(), score.shape)
    pt = (pt[1], pt[0])

    print(pt)

    cv2.rectangle(base_image, (pt[0], pt[1]), (pt[0] + wt, pt[1] + ht), (0, 0, 200), 3)
    plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()
