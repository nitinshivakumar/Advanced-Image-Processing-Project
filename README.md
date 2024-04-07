# Advanced Panorama Construction Project

## Project Overview
This project contains Python code for two advanced tasks related to image processing and panorama construction. The tasks are implemented in `task1.py` and `task2.py` files.

### Task 1: Facial Recognition
- File: `task1.py`
- Description: This file contains code for detecting and recognizing faces in input images using advanced computer vision techniques.
- Input: Images in RGB format (H x W x 3)
- Output: Detected bounding boxes of faces with recognition data in JSON format

### Task 2: Panorama Construction
- File: `task2.py`
- Description: This file contains code for stitching multiple images together to create a panoramic view using advanced image stitching algorithms.
- Input: Multiple images for panorama construction
- Output: Overlap relations as a one-hot array and the final stitched panorama image

## Folder Structure
- `__pycache__`: Cache folder
- `ComputeFBeta`: Function for computing F-beta score
- `data`: Input data folder (contains images for facial recognition and panorama construction)
- `pack_submission.sh`: Shell script for packing submission files
- `result_task1_val.json`: Result JSON file for task 1 validation
- `result_task1.json`: Result JSON file for task 1
- `task1.py`: Code file for task 1 (facial recognition)
- `task2_overlap.txt`: Output file for task 2 overlap results
- `task2_result.png`: Output file for task 2 panorama image
- `task2.py`: Code file for task 2 (panorama construction)
- `utils.py`: Utility functions file

## Usage
1. **Task 1 (Facial Recognition):**

- Replace `<input_folder_path>` with the path to the folder containing input images.
- Replace `<output_json_path>` with the desired output JSON file path.

2. **Task 2 (Panorama Construction):**

- Replace `<input_folder_path>` with the path to the folder containing images for panorama construction.
- Replace `<overlap_output_path>` with the path to save the overlap results.
- Replace `<panorama_output_path>` with the path to save the final panorama image.

## Notes
- Ensure all required libraries (`opencv-python`, `numpy`, `matplotlib`) are installed.
- Follow the project guidelines and comments in the code for proper usage.
- Debugging images can be displayed using the `show_image()` function in `utils.py`.
- Do not save intermediate files in the final submission.


