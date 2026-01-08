# sobel enlarge and ground truth verification
import numpy as np
import cv2

# Enlarge Image
def enlarge_with_edges(img, edge_map, new_h,new_w):
    
    edges_large = cv2.resize(edge_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
  
    img_linear = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    img_nearest = cv2.resize(img, (new_w, new_h),  interpolation=cv2.INTER_LANCZOS4)
    
    
    kernel = np.ones((5, 5), np.uint8) #dilation
    edge_zone = cv2.dilate(edges_large, kernel, iterations=1)
    
    weight = edge_zone.astype(np.float32) / 255.0
    
    
    if len(img.shape) == 3:  # for Color image
        weight = weight[:, :, np.newaxis]
    
    result = (weight * img_nearest + (1 - weight) * img_linear).astype(np.uint8)
        
    return result, img_linear, img_nearest

def conv(image, kernel):
    h, w, c = image.shape
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    output = np.zeros((h, w, c), dtype=np.float64)
    
    for i in range(h):  # Fixed: start from 0
        for j in range(w):  # Fixed: start from 0
            neighborhood = padded[i:i+3, j:j+3, :]
            output[i, j, :] = np.sum(neighborhood * kernel[..., np.newaxis], axis=(0, 1))
    
    return output


# Otsu's threshold
def otsu_threshold(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    
    cumsum = np.cumsum(hist)
    cumsum_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumsum_mean[-1]
    
    max_variance = 0
    threshold = 0
    
    for t in range(256):
        w0 = cumsum[t]
        w1 = 1 - w0
        
        if w0 == 0 or w1 == 0:
            continue
        
        mean0 = cumsum_mean[t] / w0
        mean1 = (global_mean - cumsum_mean[t]) / w1
        
        variance = w0 * w1 * (mean0 - mean1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    return threshold

# error calculation
def mse_loss(enl_img, orig_img):
    
    return np.mean((enl_img - orig_img) ** 2)


img = cv2.imread(r"D:\7_journal\CVML\standard_test_images\test image_created\lena_color_256_convert.tiff", 1)
gt = cv2.imread(r"D:\7_journal\CVML\standard_test_images\lena_color_512.tif", 1)

#scale = int(input("Enter Scalling Factor: "))

# Sobel kernels
sx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float64)

sy = np.array([[-1, -2, -1],
               [ 0, 0,  0],
               [ 1, 2,  1]], dtype=np.float64)

gx = conv(img, sx)
gy = conv(img, sy)


mag = np.sqrt(gx**2 + gy**2)

mag_combined = np.max(mag, axis=2)
mag_combined = cv2.convertScaleAbs(mag_combined)

th = otsu_threshold(mag_combined)
print("Otsu's threshold:",th)

imge = (mag_combined > th).astype(np.uint8) * 255

enlarged, linear, lanczos = enlarge_with_edges(img, imge, 512,512)

enl_loss=mse_loss(enlarged,gt)
lan_loss=mse_loss(lanczos,gt)

print("Final Loss: ", enl_loss)
print("Lanc Loss: ", lan_loss)

# outpath = "D:/7_journal/CVML/standard_test_images/test image_created/generated/lena_from_32.tif" 
# cv2.imwrite(outpath, enlarged)

# Display results
cv2.imshow("Ground Truth", gt)
cv2.waitKey(0)
cv2.imshow("Enlarged", enlarged)
cv2.waitKey(0)
cv2.destroyAllWindows()

