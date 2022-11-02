# Overall plan

## 1) Visualize input data
- load and plot input images (scan, segmentation mask)

## 2) Approximate vessels with circular/eliptic shape
- distinguish objects (vessels, carcinoma)
- approximate contours - circle / ellipse

## 3) Connect vessels with tangents and center-center lines
- draw lines between vessels both on left and right side
- lines: center-to-center, internal tangent, external tangent

## 4) Examine if carcinoma reaches behind each line
- detect which lines are crossed

## 5) Convert to Knosp score
- convert the rules from 4) to number on Knosp score

___

# Current steps

- better approximation of vessels - elipses / keep original shape?
- add tangent lines
- check irregularities in masks (vessels touching each other etc.)
- labelling - label only vessels (carcinoma is identified by different mask value)