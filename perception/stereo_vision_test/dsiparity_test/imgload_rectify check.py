import os
import cv2
import numpy as np

# === Blurriness Metrics ===
def compute_laplacian_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def compute_tenengrad_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(gx**2 + gy**2)
    return np.mean(g)

def compute_fft_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)

    h, w = mag.shape
    center = mag[h//4:3*h//4, w//4:3*w//4]
    total_energy = np.sum(mag)
    low_freq_energy = np.sum(center)
    high_freq_energy = total_energy - low_freq_energy
    return high_freq_energy / total_energy

# === Feature Matching with AKAZE ===
def match_features(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img

# === PATH SETUP ===
image_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
left_img_path = os.path.join(image_dir, "left_stereo/left_000.png")
right_img_path = os.path.join(image_dir, "right_stereo/right_000.png")

# === LOAD IMAGES ===
left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

if left_img is None or right_img is None:
    print("Failed to load one or both images.")
    exit()

# === BLURRINESS METRICS ===
lap_left = compute_laplacian_blur(left_img)
lap_right = compute_laplacian_blur(right_img)
tenengrad_left = compute_tenengrad_blur(left_img)
tenengrad_right = compute_tenengrad_blur(right_img)
fft_left = compute_fft_blur(left_img)
fft_right = compute_fft_blur(right_img)

print("==== Blur Metrics ====")
print(f"Laplacian  - Left: {lap_left:.2f} | Right: {lap_right:.2f}")
print(f"Tenengrad  - Left: {tenengrad_left:.2f} | Right: {tenengrad_right:.2f}")
print(f"FFT Ratio  - Left: {fft_left:.2f} | Right: {fft_right:.2f}")

# === FEATURE MATCHING VISUALIZATION ===
matched = match_features(left_img, right_img)
cv2.imshow("AKAZE Feature Matches", matched)
cv2.waitKey(0)
cv2.destroyAllWindows()
