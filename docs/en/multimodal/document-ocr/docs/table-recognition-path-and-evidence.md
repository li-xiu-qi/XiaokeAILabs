# One table: from text and structure to a populated result

This guide brings together records from two **independent runs** to show what a complete table-parsing pipeline still needs. It does not claim the models are connected, or provide an accuracy or speed comparison.

## Input and existing outputs

Both experiments saved the same self-drawn 1000 × 500 source image, with SHA-256 `3bc084da2c422c23b7c04a1bdc63078e0e5ef363420323065ee1a2189d0b1802`. It has 4 rows, 3 columns and 12 cells, without merged cells, handwriting or rotated text.

| Stage | Actual input | Saved output | Observation |
| --- | --- | --- | --- |
| Text detection and recognition | Full source image | Text polygons, recognized text, and scores | Text in all 12 detected regions matches the known content; [original record](../../../../../experiments/paddle-ocr-detection-recognition/results/recognition.json). |
| Table-structure prediction | Original image and manually cropped table | HTML-like structure tokens and score | The full image produced an extra row; the crop yielded 4 rows of 3 cells; [crop](../../../../../experiments/table-structure-recognition/results/cropped.json) · [full image](../../../../../experiments/table-structure-recognition/results/uncropped.json). |

The crop corresponds to `(40, 100, 951, 401)` in the original image, so its origin differs. A future implementation needs to add that offset when aligning crop coordinates with original-image coordinates, and undo any additional scaling or rotation. Correct coordinate alignment alone does not establish correct cell assignment.

## What is missing

The OCR output contains polygons and text such as `Item`; a `<td></td>` structure token represents an empty cell and contains no recognized text. Producing a populated table requires usable cell locations, followed by assigning text regions to cells and handling multi-line cells, spans, and boundary overlaps. The existing scripts and records **do not perform or evaluate this step**, nor do they establish that the structure output includes sufficient location data for matching.

A complete page also needs table-region detection. Here, the structure experiment uses a manually specified crop, not automatic detection. Borderless tables, skew, blur, and merged cells are not covered by this single synthetic image.

## Evaluating the complete path

Use documents annotated with text locations, transcriptions, and cell relationships. Evaluate detection and recognition separately, then evaluate the reconstructed table's contents and structure. Preserve the common input, model versions, coordinate transformations, intermediate outputs, and assembled result. Distinguish model confidence from accuracy calculated against annotations. This makes it possible to locate errors in detection, recognition, structure prediction, or matching.

[Series overview](../README.md)
