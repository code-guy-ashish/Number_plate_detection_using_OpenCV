import cv2
import imutils
import matplotlib.pyplot as plt
import easyocr
import numpy as np

img = cv2.imread('Car Image Data/download.jpg')
# cv2.imshow('Car Image', img)

# Converting the image into a grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Image', gray)

# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

# Filtering to clear out the noise in the image
img_filter = cv2.bilateralFilter(gray, 11, 17, 17)
# cv2.imshow('Filtered Image', img_filter)

# Edge detection
edges = cv2.Canny(img_filter, 30, 200)
# cv2.imshow('Canny Image', edges)

# Finding Contours
location = None
keys = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keys)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for cont in contours:
    approx = cv2.approxPolyDP(cont, 10, True)
    if len(approx) == 4:
        location = approx
        break
print(location)
# Masking
blank = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(blank, [location], 0, 255, -1)
new_img = cv2.bitwise_and(img, img, mask=blank)
# cv2.imshow('Masked Image', new_img)

# Cropping the Gray image for just number plate

(x, y) = np.where(blank == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

crop_image = gray[x1:x2 + 2, y1:y2 + 2]
# cv2.imshow('Cropped Image', crop_image)

# Using easyOCR to read the number plate
reader = easyocr.Reader(['en'])
result = reader.readtext(crop_image)
#print(result)

# Putting result on the original image
text = result[0][-2]
#print(text)
res = cv2.putText(img, text, (approx[0][0][0], approx[1][0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
res = cv2.putText(img, f'{int(result[0][-1]*100)}%', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
cv2.imshow('Result', res)

cv2.waitKey(0)
