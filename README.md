# Paragraph extraction program

This project applies digital image processing techniques to automatically extract text paragraphs from scanned or converted scientific paper images using OpenCV and NumPy. This project demonstrates practical applications of histogram projection and morphological filtering to structure unformatted scientific document images into meaningful text regions.

## Assignment Task

(i) identify and extract all the paragraphs from the papers he collected, and 
(ii) sort them in the correct order 

## What I learned

Through this project, I gained practical experience in applying document image analysis techniques to extract structured text regions from unformatted scientific paper images. Some key takeaways include:

1) Image Preprocessing for Text Extraction

  • Learned how to convert grayscale images into binary form using Otsu’s thresholding.

  • Understood how binarization simplifies further processing and separates text from background noise.

2) Connected Component Analysis

  • Used cv2.connectedComponentsWithStats() to detect and filter out large non-text components (e.g., tables, figures, images).

  • Realized how component area thresholds can effectively distinguish between body text and graphical elements.

3) Column Segmentation with Projection Profiles

  • Applied vertical projections to detect multi-column layouts in scientific papers.

  • Gained insight into how projection-based techniques help identify structured document layouts.

4) Paragraph Detection Using Horizontal Spacing

  • Implemented horizontal projections to detect paragraph boundaries based on blank rows and spacing.

  • Learned how spacing thresholds can be tuned to distinguish between lines of text and actual paragraph breaks.

5) Verification and Export of Paragraphs

  • Designed heuristics (is_text_block) to ensure detected blocks resemble real paragraphs.

  • Added padding and systematic naming when saving cropped paragraphs for better readability and usability.

6) Document Processing Challenges

  • Observed how variations in scan quality, fonts, and layouts affect segmentation accuracy.

  • Understood the importance of balancing thresholds to avoid losing small text or misclassifying large text blocks.

Overall, this project helped me connect theory from digital image processing (thresholding, connected components, projection profiles, morphological filtering) with practical challenges in document layout analysis and OCR preprocessing.
