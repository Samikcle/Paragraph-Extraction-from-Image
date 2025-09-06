import cv2
import numpy as np

# List of input image files (converted from scientific papers)
image_files = ["001.png","002.png","003.png","004.png","005.png","006.png","007.png","008.png"]

# Removes large images and tables from binary image
def remove_large_components(binary_img, area_threshold=500):
    # Find connected components 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8, ltype=cv2.CV_32S
    )
    
    # Create mask to retain small components (text)
    mask = np.zeros_like(binary_img, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= area_threshold:
            mask[labels == i] = 255
    return mask

# Saves a paragraph image to disk with padding and proper naming
def save_paragraph(para_img, base_name, para_count):
    para_img_inv = 255 - para_img
    
    # Padding values
    padding = 35  
    
    # Add padding 
    para_img_padded = cv2.copyMakeBorder(para_img_inv,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=255)
    
    # Construct output filename and save image
    output_path = base_name + "_paragraph_" + str(para_count) + ".png"
    print(output_path)
    cv2.imwrite(output_path, para_img_padded)
    return output_path

# Determines if a cropped region is likely a paragraph based on spacing
def is_text_block(para_img, spacing_threshold=0.1):
    row_sums = np.sum(para_img, axis=1)
    blank_rows = np.sum(row_sums == 0)
    spacing_ratio = blank_rows / para_img.shape[0]
    return spacing_ratio >= spacing_threshold

# Main function to extract paragraphs from a given image
def extract_paragraphs(image_path):
    min_column_width=50
    min_paragraph_height=10
    
    # Get base name of image for saving paragraph outputs
    image_basename = image_path.split('/')[-1].split('\\')[-1].split('.')[0]
    print(f"\nExtracted paragraphs from {image_basename}:")

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return []

    # Convert to binary image 
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Removes image and table for column detection
    cleaned_binary = remove_large_components(binary)
    
    # Column Segmentation
    vertical_projection = np.sum(cleaned_binary, axis=0)  # Sum of pixels in each column
    columns = []
    start_col = None

    # Detect column regions based on continuous vertical pixel density
    for i, val in enumerate(vertical_projection):
        if val > 0 and start_col is None:
            start_col = i
        elif val == 0 and start_col is not None:
            if i - start_col > min_column_width:
                columns.append((start_col, i))
            start_col = None
            
    if start_col is not None and binary.shape[1] - start_col > min_column_width:
        columns.append((start_col, binary.shape[1]))
    
    # If no columns found, treat whole image as a single column
    if not columns:
        columns = [(0, binary.shape[1])]

    para_count = 0
    saved_paths = []

    # Paragraph extraction
    for col_idx, (start_col, end_col) in enumerate(columns):
        col_img = binary[:, start_col:end_col]
        horizontal_projection = np.sum(col_img, axis=1)

        start_row = None
        blank_threshold = 25  # Number of blank rows considered as paragraph gap
        blank_count = 0
        
        # Scan rows to identify paragraph blocks
        for i, val in enumerate(horizontal_projection):
            if val > 0:
                if start_row is None:
                    start_row = i
                blank_count = 0  # Reset blank row counter
            elif start_row is not None:
                blank_count += 1
                if blank_count >= blank_threshold:
                    end_row = i - blank_count
                    if end_row - start_row > min_paragraph_height:
                        para_img = col_img[start_row:end_row, :]  # Extract paragraph
                        
                        # Verify that the block resembles a paragraph
                        if is_text_block(para_img):
                            para_count += 1
                            output_path = save_paragraph(para_img, image_basename, para_count)
                            saved_paths.append(output_path)
                    
                    # Reset for next paragraph
                    start_row = None
                    blank_count = 0

# Run the extraction for each image file
for image in image_files:
    results = extract_paragraphs(image)



