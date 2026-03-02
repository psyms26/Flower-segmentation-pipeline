import cv2
import numpy as np
import os
from pathlib import Path

# Define directories to use
# TODO - CHANGE "input_base_dir" & "ground_truth_folder"
#  TO MATCH YOUR OWN DATASET NAMES
input_base_dir = "dataset/input_images"
output_base_dir = "output"
pipeline_base_dir = "image-processing-pipeline"
ground_truth_folder = "dataset/ground_truths"

# Create output directories if they don't exist
for difficulty in ["easy", "medium", "hard"]:
    Path(os.path.join(output_base_dir, difficulty)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(pipeline_base_dir, difficulty)).mkdir(parents=True, exist_ok=True)

# Helper method for IoU prediction to truth comparison aspect
# Returns binary mask of a ground truth
def convert_gt_to_binary_mask(gt_img):
    # Convert to HSV for better color thresholding
    hsv = cv2.cvtColor(gt_img, cv2.COLOR_BGR2HSV)

    # RED mask (covers both red hue ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # BLACK mask (very low V and S)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Combine red and black as foreground
    foreground_mask = cv2.bitwise_or(red_mask, black_mask)

    binary_mask = (foreground_mask > 0).astype(np.uint8)
    return binary_mask

# Helper method to compute IoU between predicted binary mask and ground truth binary mask
# Returns IoU result
def calculate_iou(pred_mask, gt_mask):
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    return intersection / union


def apply_watershed(img, mask):
    # 1. Noise removal and sure background
    kernel = np.ones((11, 11), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # 2. Sure foreground (distance transform)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.medianBlur(mask, 11)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Adaptive thresholding for distance transform
    mean_dist = np.mean(dist_transform)
    threshold_factor = 0.2 if mean_dist > 30 else 0.25  # Adapt based on distance transform values
    ret, sure_fg = cv2.threshold(dist_transform, threshold_factor * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 3. Unknown region (edges between foreground and background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # 5. Add one to all labels so sure background is not 0
    markers = markers + 1

    # 6. Mark unknown regions as 0
    markers[unknown == 255] = 0

    # 7. Apply watershed
    image_copy = img.copy()
    markers = cv2.watershed(image_copy, markers)

    # 8. Create mask from watershed output
    final_mask = np.zeros_like(mask)
    final_mask[markers > 1] = 255

    return final_mask


# Helper method to apply a number of techniques to an input image storing each step & final output
# Returns the final mask used & segmented image result with that mask
def process_image(image_path, output_path, pipeline_dir):
    # Read the image in first
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Step 1a: Convert to best colour space
    colour_space = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(pipeline_dir, "1a_hsv_colour_space.jpg"), colour_space)

    # Step 1b: Filter out green and blue areas using HSV masking
    # HSV ranges for green with more precise filtering
    lower_green = np.array([35, 35, 35])  # Slightly relaxed lower bounds
    upper_green = np.array([85, 255, 255])

    # Create colour filtering masks
    green_mask = cv2.inRange(colour_space, lower_green, upper_green)

    # Invert mask to keep only areas that are NOT green
    keep_mask = cv2.bitwise_not(green_mask)

    # Apply the mask to HSV image
    colour_space = cv2.bitwise_and(colour_space, colour_space, mask=keep_mask)
    cv2.imwrite(os.path.join(pipeline_dir, "1b_green_filtered_hsv.jpg"), colour_space)

    # Step 2: Apply blur technique
    blurred = cv2.medianBlur(colour_space, 11)  # Increased blur size
    cv2.imwrite(os.path.join(pipeline_dir, "2_med_blur.jpg"), blurred)

    # Step 3a: Extract the best channel
    channel = blurred[:, :, 2]
    cv2.imwrite(os.path.join(pipeline_dir, "3a_extract_value_channel.jpg"), channel)

    # Step 3b: Apply median blur
    blurredM = cv2.medianBlur(channel, 11)  # Increased blur size
    cv2.imwrite(os.path.join(pipeline_dir, "3b_med_blur.jpg"), blurredM)

    # Step 3c: Apply additional smoothing
    blurredG = cv2.GaussianBlur(blurredM, (5, 5), 0)
    cv2.imwrite(os.path.join(pipeline_dir, "3c_gauss_blur.jpg"), blurredM)

    # Step 4: Apply thresholding with more aggressive parameters
    threshold = 120  # Lower threshold
    _, binary_mask = cv2.threshold(blurredG, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Additional cleaning of the binary mask
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cv2.imwrite(os.path.join(pipeline_dir, "4_thresh_mask.jpg"), binary_mask)

    # Step 5: Apply watershed
    watershed_mask = apply_watershed(image, binary_mask)
    cv2.imwrite(os.path.join(pipeline_dir, "5_watershed_mask.jpg"), watershed_mask)

    # ---- Structuring Elements ---- #
    ellipseXL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    ellipseL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    ellipseM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ellipseS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ellipseXS = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Step 6: Morphological operations
    closed = cv2.morphologyEx(watershed_mask, cv2.MORPH_CLOSE, ellipseM)  # Using larger kernel
    cv2.imwrite(os.path.join(pipeline_dir, "6a_close_morph.jpg"), closed)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ellipseXS)  # Using smaller kernel for opening
    cv2.imwrite(os.path.join(pipeline_dir, "6b_open_morph.jpg"), opened)

    mask = cv2.dilate(opened, ellipseXS, iterations=2)  # Increased iterations
    cv2.imwrite(os.path.join(pipeline_dir, "6c_dilate_morph.jpg"), mask)

    # Step 7: Apply mask to original image to get segmented flower on black background
    mask_bool = mask.astype(bool)
    segmented = np.zeros_like(image)
    segmented[mask_bool] = image[mask_bool]
    cv2.imwrite(output_path, segmented)

    return segmented, mask

# Variables for evaluation results
count = 0
total = 0
e_count = 0
e_total = 0
m_count = 0
m_total = 0
h_count = 0
h_total = 0

# Process all images by difficulty
for difficulty in ["easy", "medium", "hard"]:
    input_dir = os.path.join(input_base_dir, difficulty)
    output_dir = os.path.join(output_base_dir, difficulty)
    pipeline_difficulty_dir = os.path.join(pipeline_base_dir, difficulty)
    ground_truth_subfolder = os.path.join(ground_truth_folder, difficulty)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            pipeline_dir = os.path.join(pipeline_difficulty_dir, filename.split('.')[0])
            Path(pipeline_dir).mkdir(parents=True, exist_ok=True)

            # Construct ground truth filename (always .png)
            gt_filename = os.path.splitext(filename)[0] + ".png"
            ground_truth_path = os.path.join(ground_truth_subfolder, gt_filename)

            result, mask = process_image(image_path, output_path, pipeline_dir)
            if result is not None:
                print(f"Processed {filename} (Difficulty: {difficulty})")
            else:
                print(f"Image not found for {filename}")
                continue

            # Evaluation of the segmented image using IoU
            ground_truth_img = cv2.imread(ground_truth_path)
            if ground_truth_img is not None:
                # Resize GT to match mask
                ground_truth_img_resized = cv2.resize(ground_truth_img, (mask.shape[1], mask.shape[0]))
                gt_binary_mask = convert_gt_to_binary_mask(ground_truth_img_resized)

                # Calculate score of result & update corresponding running total & count
                iou_score = calculate_iou(mask, gt_binary_mask)
                if difficulty == "easy":
                    e_total += iou_score
                    e_count += 1
                if difficulty == "medium":
                    m_total += iou_score
                    m_count += 1
                if difficulty == "hard":
                    h_total += iou_score
                    h_count += 1
                print(f"IoU for {filename}: {iou_score:.4f}")
            else:
                print(f"Ground truth not found for {filename}")
                continue

# Compute & output evaluation results to the console
count = e_count + m_count + h_count
total = e_total + m_total + h_total
e_miou = e_total / e_count
m_miou = m_total / m_count
h_miou = h_total / h_count
overall_miou = total / count
print(f"Easy difficulty MIoU = {e_miou:.4f},")
print(f"Medium difficulty MIoU = {m_miou:.4f},")
print(f"Hard difficulty MIoU = {h_miou:.4f},")
print(f"Overall MIoU = {overall_miou:.4f}, count = {count}")