# FastROI

Utility used to select the Region Of Interest (ROI) from an image.

## Instructions

1. Copy the file paths relative to this folder and add them to `loadfiles.js`
1. Start a simple HTTP server using the parent folder as the root, for example: `python 3 -m http.server 8080`
1. Browse to the HTML GUI by navigating to `http://localhost:8080/fastroi/index.html`
1. Select the center of the desired ROI region, then navigate to the next image
1. Repeat until all images have been reviewed
1. Click `Convert`, the clipboard now contains the ROI for all images in CSV format
1. Paste the contents of the clipboard into `dataset/drawing_roi.csv`

