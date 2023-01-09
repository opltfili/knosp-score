import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from src.methods import classify_vessels, find_lines, classify_carcinoma
from src.visualization import save_layer_img

def process_scan(in_folder: str, out_folder: str) -> None:
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    
    img_path = os.path.join(in_folder, "COR_T1_C.nii")
    mask_path = os.path.join(in_folder, "mask.nii")

    # Load data with NiBabel
    img_data = nib.load(img_path)
    img = img_data.get_fdata()
    mask_data = nib.load(mask_path)
    mask = mask_data.get_fdata()

    num_layers = img.shape[2]
    scores = []

    # For each layer
    for layer in range(num_layers):
        img_slice = img[:,:,layer].T
        mask_slice = mask[:,:,layer].T

        vessels = (mask_slice==2).astype(int)
        carcinoma = (mask_slice==1).astype(int)

        classified_vessels = classify_vessels(vessels)

        if classified_vessels is None:
            continue
    
        LT, LB, RT, RB = classified_vessels
        left_lines = find_lines(LT, LB)
        right_lines = find_lines(RB, RT)

        score_l, score_r, carc_mask = classify_carcinoma(carcinoma, left_lines, right_lines)
        scores.append([layer, score_l, score_r])

        out_name = os.path.join(out_folder, "layer_{:0>2d}.png".format(layer))
        save_layer_img(out_name, img_slice, vessels, carc_mask, left_lines, right_lines,
                        layer, score_l, score_r)
    
    scores = np.array(scores)
    
    # Export csv
    csv_name = os.path.join(out_folder, "knosp.csv")
    for i in range(len(scores)+1):
        if i==0:
            with open(csv_name, 'w', newline='') as f:
                wrt = csv.writer(f)
                wrt.writerow(["layer", "left", "right"])
                f.close()
        else:
            with open(csv_name, 'a', newline='') as f:
                wrt = csv.writer(f)
                wrt.writerow(scores[i-1,:])
                f.close()
    with open(csv_name, 'a', newline='') as f:
        wrt = csv.writer(f)
        wrt.writerow(["overall"]+list(np.max(scores[:,1:], axis=0)))
        f.close()
