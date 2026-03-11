# Flower Segmentation Pipeline

This repository contains a Python-based image processing pipeline designed to segment flowers from their backgrounds in images. The pipeline leverages techniques from the OpenCV library, including color space conversion, filtering, thresholding, morphological operations, and the watershed algorithm to generate precise masks for each flower.

The project also includes a built-in evaluation system that calculates the Intersection over Union (IoU) score for each segmented image by comparing the generated mask against a ground truth, providing a quantitative measure of the pipeline's performance.



## How It Works

The segmentation process for each image involves several steps. Intermediate results of each step are saved to the `image-processing-pipeline` directory, allowing detailed inspection and debugging.

The pipeline applies the following sequence of operations:

1. **Color Space Conversion**  
   Converts the input image (BGR) to HSV (Hue, Saturation, Value), which is more effective for color-based segmentation.

2. **Green Filtering**  
   Creates a mask to filter out green areas (leaves and stems), isolating the flower.

3. **Blurring**  
   Applies a Median Blur to the filtered HSV image to reduce noise.

4. **Channel Extraction**  
   Extracts the Value (V) channel from the blurred HSV image for better contrast.

5. **Further Smoothing**  
   Applies additional Median and Gaussian blurring to reduce noise and smooth the image.

6. **Thresholding**  
   Uses Otsu's thresholding to create a binary mask, followed by morphological closing and opening for cleanup.

7. **Watershed Algorithm**  
   Refines segmentation and separates overlapping objects, especially for complex flower shapes.

8. **Morphological Refinement**  
   Applies closing, opening, and dilation to fill small holes and smooth edges.

9. **Segmentation**  
   Applies the final mask to the original image to produce the segmented flower on a black background.



## Directory Structure

- **FlowerSegmentationPipeline.py**: Main Python script that runs the pipeline.  
- **image-processing-pipeline/**: Stores intermediate images for debugging.  
- **easy/**, **medium/**, **hard/**: Subdirectories corresponding to image difficulty levels.  
- **output/**: Contains final segmented images, sorted by difficulty.  
- **Dataset (User-Provided)**:  
  - `dataset/input_images/` → Original images (`easy/`, `medium/`, `hard/`)  
  - `dataset/ground_truths/` → Ground truth masks (`easy/`, `medium/`, `hard/`)  



## Getting Started

### Prerequisites

- Python 3
- Python libraries:
  - `opencv-python`
  - `numpy`

### Installation

Clone the repository:

```bash
git clone https://github.com/psyms26/Flower-segmentation-pipeline.git
cd Flower-segmentation-pipeline
