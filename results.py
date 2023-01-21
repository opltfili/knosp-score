import numpy as np
import os
import pandas as pd
import yaml
from yaml.loader import SafeLoader

data_folder = "data"
out_folder = "output"

all_subjects = pd.DataFrame(columns=['subject', 'layer', 'L', 'pred_L', 'R', 'pred_R'])

subjects = next(os.walk(data_folder))[1]
for subject in subjects:
    in_path = os.path.join(data_folder, subject)
    out_path = os.path.join(out_folder, subject)

    with open(in_path+'/knosp.yaml', 'r') as f:
        in_yaml = yaml.load(f, Loader=SafeLoader)
    f.close()

    in_score = pd.DataFrame.from_dict(in_yaml, orient='index')
    in_score = in_score.rename({'knosp_left': 'L', 'knosp_right': 'R'}, axis='columns')
    in_score['layer'] = in_score.index.to_numpy(dtype=float)
    in_score['subject'] = int(subject)

    out_score = pd.read_csv(out_path+'/knosp.csv')
    out_score = out_score.rename({'left': 'pred_L', 'right': 'pred_R'}, axis='columns')
    out_score = out_score[out_score.layer != 'overall']
    out_score['layer'] = out_score['layer'].to_numpy(dtype=float)
    out_score['subject'] = int(subject)

    both = pd.merge(in_score, out_score,  how='outer', left_on=['subject','layer'], right_on = ['subject','layer'])
    both = both[['subject', 'layer', 'L', 'pred_L', 'R', 'pred_R']]
    
    all_subjects = pd.concat((all_subjects, both))
    all_subjects = all_subjects.replace({3.1: 3, 3.2: 3, 3.3: 3})
    all_subjects = all_subjects.replace({'L': {5: 4}, 'R': {5: 4}})

dif = all_subjects[all_subjects['L'] != all_subjects['pred_L']]
rows = ((all_subjects['R'].to_numpy() != all_subjects['pred_R'].to_numpy()).astype(int) *
        (all_subjects['L'].to_numpy() == all_subjects['pred_L'].to_numpy()).astype(int)).astype(bool)
dif = pd.concat((dif, all_subjects[rows]))
dif['dif_L'] = dif['pred_L'].to_numpy() - dif['L'].to_numpy()
dif['dif_R'] = dif['pred_R'].to_numpy() - dif['R'].to_numpy()
dif = dif.sort_values(by = ['subject', 'layer'])

all_subjects.to_csv(out_folder+"/all.csv", index=False)
dif.to_csv(out_folder+"/difference.csv", index=False)

L = all_subjects['L'].to_numpy()
R = all_subjects['R'].to_numpy()
pred_L = all_subjects['pred_L'].to_numpy()
pred_R = all_subjects['pred_R'].to_numpy()

n = len(L)
same_L = (L == pred_L).sum()
same_R = (R == pred_R).sum()
dif1_L = (np.abs(L - pred_L) <= 1).sum()
dif1_R = (np.abs(R - pred_R) <= 1).sum()

print("Accuracy report")
print("- left side:")
print("   - exact match in classification:  {:d} layers out of {:d}, that is {:6.2f}%".format(same_L, n, 100*same_L/n))
print("   - within difference of +-1 grade: {:d} layers out of {:d}, that is {:6.2f}%".format(dif1_L, n, 100*dif1_L/n))
print("- right side:")
print("   - exact match in classification:  {:d} layers out of {:d}, that is {:6.2f}%".format(same_R, n, 100*same_R/n))
print("   - within difference of +-1 grade: {:d} layers out of {:d}, that is {:6.2f}%".format(dif1_R, n, 100*dif1_R/n))
print("- both sides combined:")
print("   - exact match in classification:  {:6.2f}%".format(100*(same_L+same_R)/(2*n)))
print("   - within difference of +-1 grade: {:6.2f}%".format(100*(dif1_L+dif1_R)/(2*n)))
print("")
print("All classifications stored to "+out_folder+"/all.csv, all different classifications to "+out_folder+"/difference.csv.")
