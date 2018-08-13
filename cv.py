import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('org-1.jpg')


def ancuti_journal(im):
    # img1 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = im.copy()
    avg_b = np.average(img[:, :, 0])
    avg_g = np.average(img[:, :, 1])
    avg_r = np.average(img[:, :, 2])
    # print (avg_b)
    # print (avg_g)
    # print (avg_r)
    img[:, :, 2] = img[:, :, 2] + (1 * (avg_g - avg_r) * (1 - img[:, :, 2] / 255) * img[:, :, 1] / 255)
    # img[:, :, 0] = img[:, :, 0] + (1 * (avg_g - avg_b) * (np.sum(img[:, :, 0]) - img[:, :, 0]) * img[:, :, 1])
    img[:, :, 0] = img[:, :, 0] + (1 * (avg_g - avg_b) * (1 - img[:, :, 0] / 255) * img[:, :, 1] / 255)

    # avg_b = np.average(img[:, :, 0])
    # avg_g= np.average(img[:, :, 1])
    # avg_r = np.average(img[:, :, 2])
    #
    # print (avg_b)
    # print (avg_g)
    # print (avg_r)

    return img


def white_balance_with_ancuti(img):
    img_red_blue_compensated = ancuti_journal(img)
    result = grey_world_assumption(img_red_blue_compensated)
    return result


def grey_world_assumption(img):
    # img_red_compensated = ancuti_journal(img)
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def adjust_gamma(image, gamma):
    img = image.copy()
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)


def unsharp_masking(img):
    image = img.copy()
    # norm_mask = image.copy()

    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)  # taking the blur image
    g_mask = cv2.addWeighted(image, 1, gaussian_blur, -1, 0)  # subtracting from image to achieve mask

    # histrogram stretching of the mask
    img_yuv = cv2.cvtColor(g_mask, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    g_mask_norm = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # cv2.normalize(g_mask, norm_mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.addWeighted(image, 1, g_mask_norm, .15, 0)  # adding to initial image to get masked data

    # as per paper, this normalized addition should again be divided by 2(but that seems a loss to me, so avoiding it for now)

    # image = cv2.addWeighted(img, 1, g_mask, 1, 0)
    # cv2.normalize(image, image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # return unsharp_image

    #
    # f = np.hstack((img, image, g_mask, norm_image))
    # plt.imshow(f)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    return image


def laplacian_weight(img):
    image = img.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Laplacian(image, cv2.CV_16S)
    cv2.convertScaleAbs(image, image)
    # image = cv2.addWeighted(image, 1, img, -1, 0)
    return image


def saliency_weight(img):
    image = img.copy()

    return image


def saturation_weight(img):
    image = img.copy()
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lab_image[:, :, 0] = (((image[:, :, 0] - lab_image[:, :, 0]) ** 2 + (image[:, :, 1] - lab_image[:, :, 0]) ** 2 + (
                image[:, :, 2] - lab_image[:, :, 0]) ** 2) * (0.33)) ** (.5)
    image = cv2.cvtColor(lab_image, cv2.COLOR_YCrCb2BGR)
    return image


white_balanced = white_balance_with_ancuti(img)
gamma_adjusted = adjust_gamma(white_balanced, gamma=0.5)
unsharp_masked = unsharp_masking(white_balanced)
final = np.hstack((white_balanced, unsharp_masked, gamma_adjusted, saturation_weight(unsharp_masked)))
# final = white_balance(img)


# cv2.imshow("WhiteBalanced",white_balance(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

final = final[:, :, ::-1]
plt.imshow(final)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# img[:, :, 0] = 0  #setting green channel to zero
# img[:, :, 1] = 0


# b,g,r = cv2.split(img)
# img2 = cv2.merge([r,g,b])
# img2 = img[:,:,::-1]

# print(img2.shape)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.subplot(121);plt.imshow(img) # expects distorted color
# plt.subplot(122);plt.imshow(img2) # expect true color
# plt.show()
