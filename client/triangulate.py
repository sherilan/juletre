import pathlib 
import os 
import itertools 
import numpy as np 
import pandas as pd 

import camera 
import utils 

folder = pathlib.Path(__file__).parent / 'data'
tree = 'tree1'
ipt_folder = folder / 'preprocessing' / tree

views = [
    {
        'file': ipt_folder / '1.csv',
        'angle': 0 * np.pi / 2
    },
    {
        'file': ipt_folder / '2.csv',
        'angle': 1 * np.pi / 2
    },
    {
        'file': ipt_folder / '3.csv',
        'angle': 2 * np.pi / 2
    },
    {
        'file': ipt_folder / '4.csv',
        'angle': 3 * np.pi / 2
    },
]
n_views = len(views)
threshold = 50

DIST = 3.52 
HEIGHT = 0.98
K = camera.Camera.Intrinsics.macbook_pro_2020().mtx

dfs = [pd.read_csv(v['file'], index_col=0) for v in views]

assert all((df.index == dfs[0].index).all() for df in dfs)
n_leds = len(dfs[0])
index = dfs[0].index
ok = np.stack([df.v.values > threshold for df in dfs], 1)
uv = np.stack([df[['x', 'y']].values for df in dfs], 1)
Rs, ts, Ps = [], [], []

for i, v in enumerate(views):
    R, t = utils.gen_extrinsics(dist=DIST, height=HEIGHT, angle=v['angle'])
    P = utils.get_projection_matrix(K, R, t)
    Rs.append(R)
    ts.append(t)
    Ps.append(P)

points = []
for i in range(n_leds):
    _Ps, _ps = [], [] 
    for j in range(n_views):
        if not ok[i, j]:
            continue 
        _Ps.append(Ps[j])
        _ps.append(uv[i, j])
    if len(_Ps) < 2:
        continue 
    
    set_indices = list(range(len(_ps)))
    min_resid = np.inf 
    for k in range(2, len(_Ps) + 1):
        for subset in itertools.combinations(set_indices, k):
            (x, y, z), resid = utils.traingulate_multi(
                ps=[_ps[j] for j in subset], 
                Ps=[_Ps[j] for j in subset]
            )
            r = resid[0]/k**2 if len(resid) else float('inf')
            if r < min_resid:
                min_resid = r 
                result = {
                    'index': index[i],
                    'x': x[0],
                    'y': y[0],
                    'z': z[0],
                    'k': k,
                    'r': r,
                }

    points.append(result)

points = pd.DataFrame(points).set_index('index')
print(points.shape)
r05, r95 = points.r.quantile((0.05, 0.95))
utils.viz_pointcloud(points[list('xyz')], c=points['r'].clip(r05, r95)).show()
# print(points.describe())

print(points.reindex(index))
points.reindex(index).to_csv('points.csv')