import PIL.Image as Image
from numpy import array
from numpy import abs

import numpy
numpy.set_printoptions(threshold=numpy.nan)
from math import sqrt

CL = [(1 + sqrt(3)) / (4 * sqrt(2)),
      (3 + sqrt(3)) / (4 * sqrt(2)),
      (3 - sqrt(3)) / (4 * sqrt(2)),
      (1 - sqrt(3)) / (4 * sqrt(2))]


def hpf_coeffs(CL):
    N = len(CL)
    CH = [(-1) ** k * CL[N - k - 1]
          for k in xrange(N)]
    return CH


def pconv(data, CL, CH, delta=0):
    assert (len(CL) == len(CH))
    N = len(CL)
    M = len(data)
    out = []
    for k in xrange(0, M, 2):
        sL = 0
        sH = 0
        for i in xrange(N):
            sL += data[(k + i - delta) % M] * CL[i]
            sH += data[(k + i - delta) % M] * CH[i]
        out.append(sL)
        out.append(sH)
    return out


def dwt2(image, CL):
    CH = hpf_coeffs(CL)
    w, h = image.shape
    imageT = image.copy()
    for i in xrange(h):
        imageT[i, :] = pconv(imageT[i, :], CL, CH)
    for i in xrange(w):
        imageT[:, i] = pconv(imageT[:, i], CL, CH)

    data = imageT.copy()
    data[0:h / 2, 0:w / 2] = imageT[0:h:2, 0:w:2]
    data[h / 2:h, 0:w / 2] = imageT[1:h:2, 0:w:2]
    data[0:h / 2, w / 2:w] = imageT[0:h:2, 1:w:2]
    data[h / 2:h, w / 2:w] = imageT[1:h:2, 1:w:2]
    return data

def icoeffs(CL, CH):
    assert(len(CL) == len(CH))
    iCL = []
    iCH = []
    for k in xrange(0, len(CL), 2):
        iCL.extend([CL[k-2], CH[k-2]])
        iCH.extend([CL[k-1], CH[k-1]])
    return (iCL, iCH)

def idwt2(data, CL):
    w, h = data.shape
    imageT = data.copy()
    imageT[0:h:2, 0:w:2] = data[0:h / 2, 0:w / 2]
    imageT[1:h:2, 0:w:2] = data[h / 2:h, 0:w / 2]
    imageT[0:h:2, 1:w:2] = data[0:h / 2, w / 2:w]
    imageT[1:h:2, 1:w:2] = data[h / 2:h, w / 2:w]

    CH = hpf_coeffs(CL)
    iCL, iCH = icoeffs(CL, CH)
    image = imageT.copy()
    for i in xrange(w):
        image[:, i] = pconv(image[:, i], iCL, iCH, delta=len(iCL) - 2)
    for i in xrange(h):
        image[i, :] = pconv(image[i, :], iCL, iCH, delta=len(iCL) - 2)

    return image


image = Image.open('sphere/pic38.png').convert('L')
image = image.resize((624, 624))

img1 = Image.open('sphere_png/pic24.png').convert('L')
img1 = image.resize((624, 624))
img2 = Image.open('sphere_png/pic25.png').convert('L')
img2 = image.resize((624, 624))
img3 = Image.open('sphere_png/pic26.png').convert('L')
img3 = image.resize((624, 624))
img4 = Image.open('sphere_png/pic27.png').convert('L')
img4 = image.resize((624, 624))
img5 = numpy.concatenate((img1, img2), axis=1)
img = numpy.concatenate((img3, img4), axis=1)
img6 = numpy.concatenate((img, img5), axis=0)
image = array(img6)
arr2 = []
data2 = dwt2(image, CL)

threshold = 100
counter = 0
all = 0
print(data2)
for i in range(0, len(data2)):
    for j in range(0, len(data2[i])):
        if data2[i][j] == 0:
            counter = counter + 1
        all = all + 1
print('All')
print(all)
print('Before')
print(counter)
counter = 0
data2[abs(data2) < threshold] = 0
for i in range(0, len(data2)):
    for j in range(0, len(data2[i])):
        if data2[i][j] == 0:
            counter = counter + 1
print('After')
print(counter)
img = Image.fromarray(data2, 'L')

image_rec = idwt2(data2, CL)
img = Image.fromarray(image_rec, 'L')
img.show()