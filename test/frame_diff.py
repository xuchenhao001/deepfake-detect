import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def show_np_images(images: list[np.ndarray]):
    plt.rcParams['figure.constrained_layout.use'] = True
    # plot original and manipulated images
    for i, image in zip(range(len(images)), images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


def simple_diff(image1: np.ndarray, image2: np.ndarray):
    image1 = cv2.medianBlur(image1, ksize=5)
    image2 = cv2.medianBlur(image2, ksize=5)
    diff_img = image2 - image1
    # threshold = 250
    # diff_img[diff_img >= threshold] = 0
    return diff_img


def euclidean_diff(image1: np.ndarray, image2: np.ndarray):
    image1_hsv = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image2_hsv = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
    # https://stackoverflow.com/a/1401828
    dist = np.linalg.norm(image2_hsv - image1_hsv, axis=2)
    print(f"Debug dist max: {dist.max()}")
    threshold = 400
    dist[dist < threshold] = 0
    return dist


def ssim_diff(image1: np.ndarray, image2: np.ndarray):
    # Structural Similarity Index (SSIM): https://stackoverflow.com/a/56193442
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Compute SSIM between the two images
    score, diff = structural_similarity(image1_gray, image2_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    threshold = 100
    diff[diff <= threshold] = 0
    diff[diff > threshold] = 255
    return diff


def test():
    ori_img_pil = Image.open("./original/frame00001.jpg").convert(mode="RGB")
    man_img_pil = Image.open("./manipulated/frame00001.jpg").convert(mode="RGB")
    ori_img = np.asarray(ori_img_pil)
    man_img = np.asarray(man_img_pil)

    # diff_img = simple_diff(ori_img, man_img)
    # diff_img = euclidean_diff(ori_img, man_img)
    diff_img = ssim_diff(ori_img, man_img)

    show_np_images([ori_img, man_img, diff_img])


if __name__ == "__main__":
    test()
