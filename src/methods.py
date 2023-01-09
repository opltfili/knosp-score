import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import label
from sklearn.cluster import KMeans
from src.objects import Vessel, Line

def classify_vessels(vessels: np.ndarray):
    pts = np.argwhere(vessels>0)
    if pts.shape[0] < 4:
        return None
            
    kmeans = KMeans(n_clusters=4)
    labels = kmeans.fit_predict(pts)

    coms = np.zeros((4,3)) # centre of mass
    for i in range(4):
        coms[i,2] = i
        coms[i,0:2] = np.mean(pts[labels==i,:], axis=0)

    middle = np.mean(coms[:, 0:2], axis=0)
    left = coms[coms[:,1]<=middle[1]]
    right = coms[coms[:,1]>middle[1]]

    assert left.shape[0] == 2 and right.shape[0] == 2, "Cannot distinguish left and right vessels."

    LT_idx = left[0,2] if left[0,0] < left[1,0] else left[1,2]
    LB_idx = left[0,2] if left[0,0] >= left[1,0] else left[1,2]
    RT_idx = right[0,2] if right[0,0] < right[1,0] else right[1,2]
    RB_idx = right[0,2] if right[0,0] >= right[1,0] else right[1,2]

    out = []
    for idx in [LT_idx, LB_idx, RT_idx, RB_idx]:
        pts_i = pts[labels==idx]
        x_i, y_i = pts_i.T
        bin_mask = np.zeros_like(vessels)
        bin_mask[x_i, y_i] = 1
        out.append(Vessel(pts_i, bin_mask, coms[coms[:,2]==idx,0:2].squeeze()))
    
    return out


def is_pt_out(pts: np.ndarray, line: Line):
    out = (pts - line.b) @ line.v < 0
    eps = np.abs((pts - line.b) @ line.v) > 0.5

    return out*eps


def find_lines(LT: Vessel, LB: Vessel):
    mid = Line(LT.com, LB.com)

    both = np.vstack((LT.pts, LB.pts))
    hull = ConvexHull(both)
    vertices = hull.points[hull.vertices]
    vertices = np.vstack((vertices, [vertices[0]])).astype(int)
    vert_id = LT.mask[vertices[:,0], vertices[:,1]] + 2*LB.mask[vertices[:,0], vertices[:,1]]
    diff = np.diff(vert_id)

    idx1 = np.argwhere(diff==-1)
    idx2 = np.argwhere(diff==1)

    if is_pt_out(vertices[idx1], mid):
        lat = Line(vertices[idx1+1].squeeze(), vertices[idx1].squeeze())
        med = Line(vertices[idx2].squeeze(), vertices[idx2+1].squeeze())
    else:
        med = Line(vertices[idx1+1].squeeze(), vertices[idx1].squeeze())
        lat = Line(vertices[idx2].squeeze(), vertices[idx2+1].squeeze())

    return [lat, mid, med]


def classify_carcinoma(carcinoma: np.ndarray, left_lines: list, right_lines: list):
    carc_pts = np.argwhere(carcinoma==1)
    l_lat, l_mid, l_med = left_lines
    r_lat, r_mid, r_med = right_lines

    # Left side
    gr1_l = is_pt_out(carc_pts, l_med)
    gr2_l = is_pt_out(carc_pts, l_mid)
    gr3_l = is_pt_out(carc_pts, l_lat)

    mask_l = 10*np.copy(carcinoma)
    mask_l[carc_pts[gr1_l,0], carc_pts[gr1_l,1]] = 11
    mask_l[carc_pts[gr2_l,0], carc_pts[gr2_l,1]] = 12
    mask_l[carc_pts[gr3_l,0], carc_pts[gr3_l,1]] = 13

    score_l = mask_l.max()-10

    if score_l == 3:
        com_y_inf = l_mid.a[0] if l_mid.a[0] > l_mid.b[0] else l_mid.b[0]
        com_y_sup = l_mid.a[0] if l_mid.a[0] < l_mid.b[0] else l_mid.b[0]
        new_l_inf = classify_grade4(mask_l, com_y_inf)
        if new_l_inf == 4.:
            score_l = 4.
        else:
            new_l_sup = classify_grade4(mask_l, com_y_sup)
            if new_l_sup == 4.:
                score_l = 4.
            else:
                score_l = new_l_inf

    # Right side
    gr1_r = is_pt_out(carc_pts, r_med)
    gr2_r = is_pt_out(carc_pts, r_mid)
    gr3_r = is_pt_out(carc_pts, r_lat)

    mask_r = 10*np.copy(carcinoma)
    mask_r[carc_pts[gr1_r,0], carc_pts[gr1_r,1]] = 11
    mask_r[carc_pts[gr2_r,0], carc_pts[gr2_r,1]] = 12
    mask_r[carc_pts[gr3_r,0], carc_pts[gr3_r,1]] = 13

    score_r = mask_r.max()-10

    if score_r == 3:
        com_y_inf = r_mid.a[0] if r_mid.a[0] > r_mid.b[0] else r_mid.b[0]
        com_y_sup = r_mid.a[0] if r_mid.a[0] < r_mid.b[0] else r_mid.b[0]
        new_r_inf = classify_grade4(mask_r, com_y_inf)
        if new_r_inf == 4.:
            score_r = 4.
        else:
            new_r_sup = classify_grade4(mask_r, com_y_sup)
            if new_r_sup == 4.:
                score_r = 4.
            else:
                score_r = new_r_inf

    # Total
    carc_mask = np.where(mask_l == 10, mask_r, mask_l)

    return score_l, score_r, carc_mask


def classify_grade4(mask: np.ndarray, com_y: float):
    mask3 = (mask==13).astype(int)
    labeled, n = label(mask3, return_num=True)
    sup = False
    inf = False

    for i in range(1, n+1):
        sup_i = False
        inf_i = False
        pxs = np.argwhere(labeled==i)
        for px in pxs:
            y, x = px
            if 12 in mask[y-1:y+2, x-1:x+2]:
                if y < com_y:
                    sup_i = True
                else:
                    inf_i = True
        if sup_i and inf_i:
            return 4.
        else:
            sup = sup or sup_i
            inf = inf or inf_i
    
    if sup and inf:
        return 3.3
    elif sup:
        return 3.1
    elif inf:
        return 3.2
    else:
        return 3.0
