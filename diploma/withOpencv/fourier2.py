import numpy as np

# np.set_printoptions(threshold=np.nan)
import cv2
import math


def createMask(h, w, d):
    # d = 5
    h0 = h / 2
    w0 = w / 2
    maskL = np.zeros((h, w, 2), np.uint8)
    percent = 0
    for i in range(0, h):
        for j in range(0, w):
            if math.hypot(i - h0, (j - w0)/3) <= d:
                maskL[i][j] = 1
                percent = percent + 1
            # coef = h/30
            # if h/2 - coef < i < h/2 + coef:
            #     if w / 2 - coef < j < w / 2 + coef:
            #         maskL[i][j] = 1
            #         percent = percent + 1
    print(percent)
    print(h * w)
    return maskL

size = 624
d = size / 15

img1 = cv2.imread('sphere/pic19.png')
img1 = cv2.resize(img1, (size, size))
img2 = cv2.imread('sphere/pic21.png')
img2 = cv2.resize(img2, (size, size))
img3 = cv2.imread('sphere/pic22.png')
img3 = cv2.resize(img3, (size, size))
img = np.concatenate((img1, img2, img3), axis=1)
rows, cols = img.shape[:2]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # сдвигаемся, чтобы в центре был центр
h, w = img.shape
magnitude_spectrum = 10 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  # чтобы немного ярче было
maskL = createMask(h, w, d)
fshiftL = dft_shift * maskL
magnitude_spectrum_l = 10 * np.log(cv2.magnitude(fshiftL[:, :, 0], fshiftL[:, :, 1]))
cv2.imshow("img", magnitude_spectrum_l)
cv2.waitKey(0)
f_l_ishift = np.fft.ifftshift(fshiftL)  # сдвигаемся обратно
img_back_l = cv2.idft(f_l_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
img_back_l = img_back_l.astype(np.uint8)

cv2.imshow("img", img_back_l)

cv2.imwrite("three_img.png", img_back_l)
cv2.waitKey(0)
