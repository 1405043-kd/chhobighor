import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('for_science4.jpg')


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


final = np.hstack((img, white_balance_with_ancuti(img), grey_world_assumption(img)))
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
