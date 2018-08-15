import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('im2.jpg')


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

    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 3)  # taking the blur image
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
    # #as per equation 7 described in achantay a
    # image = img.copy()
    # gaussian_blur = cv2.GaussianBlur(image, (3, 3), 3)
    # #image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # #image_lab_blur = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2LAB)
    #
    # avg_b = np.average(image[:, :, 0])
    # avg_g = np.average(image[:, :, 1])
    # avg_r = np.average(image[:, :, 2])
    #
    # image[:, :, 0] = abs(avg_b-gaussian_blur[:, :, 0])
    # image[:, :, 1] = abs(avg_g-gaussian_blur[:, :, 1])
    # image[:, :, 2] = abs(avg_r-gaussian_blur[:, :, 2])
    #
    # #image_lab_blur = cv2.cvtColor(gaussian_blur, cv2.COLOR_LAB2BGR)
    #
    # return image

    # as per equation 8 described in achantay et al.

    image = img.copy()

    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 3)
    # image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # image_lab_blur = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2Lab)
    image_lab = image
    image_lab_blur = gaussian_blur

    cv2.transpose(image_lab, image_lab)

    avg_l = np.average(image_lab[:, :, 0])
    avg_a = np.average(image_lab[:, :, 1])
    avg_b = np.average(image_lab[:, :, 2])

    image_lab[:, :, 0] = (abs((avg_l) - (image_lab_blur[:, :, 0])))
    image_lab[:, :, 1] = (abs((avg_a) - (image_lab_blur[:, :, 1])))
    image_lab[:, :, 2] = (abs((avg_b) - (image_lab_blur[:, :, 2])))

    # image_lab = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)

    return image_lab


def saturation_weight(img):
    image = img.copy()
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lab_image[:, :, 0] = (((image[:, :, 0] - lab_image[:, :, 0]) ** 2 + (image[:, :, 1] - lab_image[:, :, 0]) ** 2 + (
            image[:, :, 2] - lab_image[:, :, 0]) ** 2) * (0.33)) ** (.5)
    image = cv2.cvtColor(lab_image, cv2.COLOR_YCrCb2BGR)
    return image


def weight_adder(input):
    weight_sum = input.copy()
    input1 = input.copy()
    lap = laplacian_weight(input1)
    sal = saliency_weight(input1)
    sat = saturation_weight(input1)

    # weight_sum=cv2.add(lap, sal)
    # weight_sum = cv2.add(weight_sum, sat, weight_sum.type())
    #
    weight_sum[:, :, 0] = (lap[:, :, 0] + sal[:, :, 0] + sat[:, :, 0])
    weight_sum[:, :, 1] = (lap[:, :, 1] + sal[:, :, 1] + sat[:, :, 1])
    weight_sum[:, :, 2] = (lap[:, :, 2] + sal[:, :, 2] + sat[:, :, 2])

    # weight_sum=cv2.addWeighted(lap, 1.0, sal, 1.0)
    # weight_sum=cv2.addWeighted(weight_sum, 1, sat, 1)

    # cv2.divide(input1, weight_sum, input1)

    # input1[:, :, 0] = (input1[:, :, 0])
    # input1[:, :, 1] = (input1[:, :, 1])
    # input1[:, :, 2] = (input1[:, :, 2])

    # cv2.normalize(weight_sum, weight_sum, 0, 255, cv2.NORM_L1)

    return weight_sum


def adder(image1, image2):
    src = image1.copy()
    dst = image2.copy()
    # dst[:, :, 0]=dst[:, :, 0]+src[:, :, 0]
    # dst[:, :, 1]=dst[:, :, 1]+src[:, :, 1]
    # dst[:, :, 2] = dst[:, :, 2] + src[:, :, 2]
    cv2.add(src, dst, dst)
    return dst


def normal_fusion(gamma, gamma_w, unsharp, unsharp_w):
    # basic_fusion
    im1 = cv2.multiply(gamma, gamma_w)
    im2 = cv2.multiply(unsharp, unsharp_w)

    res = cv2.add(im1, im2)

    return res


def multi_fusion(gamma, gamma_w, unsharp, unsharp_w):
    # basic_fusion
    im1 = cv2.multiply(gamma, gamma_w)
    im2 = cv2.multiply(unsharp, unsharp_w)

    res = cv2.add(im1, im2)

    return res


def gaussian_pyramid():
    return


def laplacian_pyramid():
    return


white_balanced = white_balance_with_ancuti(img)
gamma_adjusted = adjust_gamma(white_balanced, gamma=0.5)
unsharp_masked = unsharp_masking(white_balanced)
image_and_white_balanced = np.hstack((img, white_balanced))
two_input = np.hstack((white_balanced, unsharp_masked, gamma_adjusted))
gamma_weights = np.hstack((gamma_adjusted, laplacian_weight(gamma_adjusted), saliency_weight(gamma_adjusted),
                           saturation_weight(gamma_adjusted)))
unsharp_weights = np.hstack((unsharp_masked, laplacian_weight(unsharp_masked), saliency_weight(unsharp_masked),
                             saturation_weight(unsharp_masked)))

weighted_gamma = weight_adder(gamma_adjusted)
weighted_unsharped = weight_adder(unsharp_masked)
added_ga_un = cv2.add(weighted_gamma, weighted_unsharped)

first_w = cv2.divide(weighted_gamma, added_ga_un)
second_w = cv2.divide(weighted_unsharped, added_ga_un)

nor_f = normal_fusion(gamma_adjusted, first_w, unsharp_masked, second_w)

weighted_singular = np.hstack((weighted_gamma, weighted_unsharped, added_ga_un))

merging_final = np.hstack((img, nor_f))

# final = white_balance(img)


# cv2.imshow("WhiteBalanced",white_balance(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_and_white_balanced = image_and_white_balanced[:, :, ::-1]
two_input = two_input[:, :, ::-1]
gamma_weights = gamma_weights[:, :, ::-1]
unsharp_weights = unsharp_weights[:, :, ::-1]
weighted_singular =weighted_singular[:, :, ::-1]
merging_final = merging_final[:, :, ::-1]


plt.figure('Step 6: merging final two')
plt.imshow(merging_final)
plt.title('Final')
plt.xticks([]), plt.yticks([])

plt.figure('Step 1: image to white balanced version')
plt.imshow(image_and_white_balanced)
plt.title('Original Image, White Balanced Imaage')
plt.xticks([]), plt.yticks([])

plt.figure('Step 2: generate two input')
plt.imshow(two_input)
plt.title('White Balanced, Unsharp Masked, Gamma Corrected')
plt.xticks([]), plt.yticks([])

plt.figure('Step 3: gamma input weighted')
plt.imshow(gamma_weights)
plt.title('Gamma Corrected, Laplacian Weight, Saliency Weight, Saturation Weight')
plt.xticks([]), plt.yticks([])

plt.figure('Step 4: unsharp input weighted')
plt.imshow(unsharp_weights)
plt.title('Unsharp Masked, Laplacian Weight, Saliency Weight, Saturation Weight')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.figure('Step 5: weighted into one image')
plt.imshow(weighted_singular)
plt.title('Gamma Weighted, Unsharp Weighted, Total Weighted')
plt.xticks([]), plt.yticks([])

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
