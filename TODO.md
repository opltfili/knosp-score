# Overall plan

## 1) Visualize input data
- [x] load and plot input images (scan, segmentation mask)

## 2) Create separate objects for vessels and carcinoma
- [x] distinguish objects (vessels, carcinoma)

## 3) Connect vessels with tangents and center-center lines
- [x] draw lines between vessels both on left and right side
- [x] lines: center-to-center, internal tangent, external tangent

## 4) Examine if carcinoma reaches behind each line
- [x] detect presence of carcinoma area behind line

## 5) Convert to Knosp score
- [x] convert the rules from 4) to number on Knosp score

___

# Current steps

- add comments and docstrings
- adjust names of variables
- check PEP8 conventions with pydocstyle
- is transpose (mask.T) correct for LR orientation? 