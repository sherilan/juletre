import pathlib 
import re 
import numpy as np 
import cv2 
import math3d as m3d 
import PIL
import seaborn as sns 
import pandas as pd 
import tqdm 

def linear_LS_triangulation(u1, P1, u2, P2):
    
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    linear_LS_triangulation_C = -np.eye(2, 3)
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u1)))
    
    # Initialize C matrices
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)
    
    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]
        
        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
        
    return x.T.astype(float), np.ones(len(u1), dtype=bool)

def triangulate(p1, p2, K, R, t):
    linear_LS_triangulation_C = -np.eye(2, 3)
    # REF: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py
    def linear_LS_triangulation(u1, P1, u2, P2):
        """
        Linear Least Squares based triangulation.
        Relative speed: 0.1
        
        (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
        (u2, P2) is the second pair.
        
        u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
        
        The status-vector will be True for all points.
        """
        A = np.zeros((4, 3))
        b = np.zeros((4, 1))
        
        # Create array of triangulated points
        x = np.zeros((3, len(u1)))
        
        # Initialize C matrices
        C1 = np.array(linear_LS_triangulation_C)
        C2 = np.array(linear_LS_triangulation_C)
        
        for i in range(len(u1)):
            # Derivation of matrices A and b:
            # for each camera following equations hold in case of perfect point matches:
            #     u.x * (P[2,:] * x)     =     P[0,:] * x
            #     u.y * (P[2,:] * x)     =     P[1,:] * x
            # and imposing the constraint:
            #     x = [x.x, x.y, x.z, 1]^T
            # yields:
            #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
            #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
            # and since we have to do this for 2 cameras, and since we imposed the constraint,
            # we have to solve 4 equations in 3 unknowns (in LS sense).

            # Build C matrices, to construct A and b in a concise way
            C1[:, 2] = u1[i, :]
            C2[:, 2] = u2[i, :]
            
            # Build A matrix:
            # [
            #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
            #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
            #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
            #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
            # ]
            A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
            A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
            
            # Build b vector:
            # [
            #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
            #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
            #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
            #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
            # ]
            b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
            b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
            b *= -1
            
            # Solve for x vector
            cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
        
        return x.T.astype(float), np.ones(len(u1), dtype=bool)

    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, t)))
    return linear_LS_triangulation(p1, P1, p2, P2)[0]

def triangulation_system_rows(p, P):
    """
    Builds a 2x3 triangulation matrix segment
    
    Args:
        p (float[2]) image point (u, v)
        P (float[3,4]) projection matrx
    """
    C = -np.eye(2, 3)
    C[:, 2] = p 
    A = C @ P[:, :3]
    b = -1 * C @ P[:, 3:]
    return A, b 


def traingulate_multi(ps, Ps):
    
    As, bs = [], []
    for p, P in zip(ps, Ps):
        A, b = triangulation_system_rows(p, P) 
        As.append(A)
        bs.append(b)
    A = np.concatenate(As)
    b = np.concatenate(bs)
    xyz, resid, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return xyz, resid 

def get_pointcloud(p1, p2, K):
    E, mask = cv2.findEssentialMat(p1, p2, K)#, method=cv2.RANSAC, prob=0.999, threshold=1)
    # print(f'{mask.mean()*100 :.1f}%')
    mask = mask.ravel().astype(bool)
    _, R, t, _ = cv2.recoverPose(E, p1, p2, K)
    xyz = triangulate(p1, p2, K, R, t)
    return xyz 

def viz_pointcloud(xyz, R=None, t=None, c=None):
    import pandas as pd 
    import plotly.express as px
    xyz = np.asarray(xyz)
    if R is not None:
        xyz = xyz @ R.T 
    if t is not None:
        xyz = xyz + t 
    df = pd.DataFrame(xyz, columns=list('XYZ'))
    if c is not None:
        if c is True:
            c = np.arange(len(xyz)) % 20
        df['C'] = c 
        color = 'C'
    else:
        color = None 
    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=color)
    return fig 


def gen_extrinsics(dist=4, height=1, angle=0):
    Rz = m3d.Orientation.new_rot_z(angle)
    Rx = m3d.Orientation.new_rot_x(-np.pi/2)
    R = (Rz * Rx).array
    sa, ca = np.sin(angle), np.cos(angle)
    r = np.array([sa * dist, -ca * dist, height])
    return R, r 

def sample_cone(h, r, n=None):
    if not n is None:
        return np.stack([sample_cone(h, r) for _ in range(n)])
    # Samples a cylinder with hacky rejection sampling 
    while True:
        x, y, z = np.random.uniform((-r, -r, 0), (r, r, h))
        if (x**2 + y**2) <= (r * (1 - z / h)) **2:
            return x, y, z
        

def project(p, h, w, imat, emat, evec):
    p_c = (p - evec) @ emat 
    z = p_c[:, 2]
    p_n = p_c / z[:, None]
    u, v, _ = (p_n @ imat.T).T
    return pd.DataFrame(np.stack([u, v, z], -1), columns=list('UVZ'))

def estimate_relpose(points_a, points_b):
    center_a = points_a.mean(0)
    center_b = points_b.mean(0)
    H = (points_a - center_a).T @ (points_b - center_b)
    U, _, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[-1] *= 1 
        R = Vh.T @ U.T
    r = center_b - R @ center_a 
    return R, r 



def render(p, h, w, imat, emat, evec, znear=0.01, r=1, s=1, c=None, points_only=False):
    # Map to camera frame 
    p_c = (p - evec) @ emat 
    # Filter out points behind near plane and divide by depth
    in_front = p_c[:, 2] > znear
    p_n = p_c[in_front]
    zorder = np.argsort(p_n[:, 2])
    p_n = p_n[zorder]
    z = p_n[:, 2]
    p_n = p_n / z[:, None]

    # return pd.DataFrame(p_n, columns=list('XYZ')).describe()
    # return pd.DataFrame((p_n @ imat.T), columns=list('UVZ')).describe()
    u, v, _ = (p_n @ imat.T).round().astype(int).T
    # inside = (0 <= u) & (u < w) & (0 <= v) & (v < h)
    if points_only:
        return pd.DataFrame(np.stack([u, v, z], -1), columns=list('UVZ'))

    if c is None:
        interp = 1 - (z - z.min()) / (z.max() - z.min())
        colors = np.full((3,), 255) * interp[:, None]
    else:
        c = np.asarray(c)
        assert len(c) == len(p)
        c = c[in_front][zorder]
        if c.ndim == 2:
            assert c.shape == (len(p), 3)
            colors = c 
        else:
            n_colors = c.max() + 1
            palette = sns.color_palette("tab10", n_colors)
            colors = (np.array(palette) * 255)[c].astype(int)


    # Paint image 
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 3] = 255
    for di in range(1-r, r):
        for dj in range(1-r, r):
            i = v + di 
            j = u + dj
            inside = (0 <= i) & (i < h) & (0 <= j) & (j < w)

            img[i[inside], j[inside], :3] = colors[inside] 
    return PIL.Image.fromarray(img)


def find_maximum(
    path, 
    i1=0,
    i2=None,
    j1=0,
    j2=None,
    color_proj=(1, 0, 0), 
    blur_size=5,
    blur_sigma=3,
    ret_image=False,
    paint_image=False,
):
    image = np.asarray(PIL.Image.open(path))
    i2 = image.shape[0] if i2 is None else i2
    j2 = image.shape[1] if j2 is None else j2 
    color_proj = np.array(color_proj) / np.sum(color_proj)
    gray = (image.astype(float) * (color_proj)).sum(-1)[i1:i2, j1:j2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_sigma)
    a_max = np.argmax(gray)
    v_max = float(gray.ravel()[a_max])
    x_max = j1 + (a_max % (j2 - j1 ))
    y_max = i1 +( a_max // (j2 - j1 ))
    if ret_image:
        if paint_image:
            cv2.circle(image, (x_max, y_max), 25, (0,0,255), 5)
        return image, x_max, y_max, v_max 
    else:
        return x_max, y_max, v_max 



def get_index_from_path(path):
    return int(re.match('img_(\d*).png', path.name).groups()[0])

def load_images(view_folder):
    view_folder = pathlib.Path(view_folder)
    paths = sorted(view_folder.iterdir(), key=get_index_from_path)
    indices = list(map(get_index_from_path, paths))
    return indices, paths 

def preprocess_data(view_folder, **kwargs):
    data = []
    view_folder = pathlib.Path(view_folder)
    paths = sorted(view_folder.iterdir(), key=get_index_from_path)
    for path in tqdm.tqdm(paths):
        d = {}
        d['path'] = path 
        d['index'] = get_index_from_path(path) 
        x_max, y_max, v_max = find_maximum(path, **kwargs)
        d['x'] = x_max 
        d['y'] = y_max 
        d['v'] = v_max 
        data.append(d)
    return pd.DataFrame(data)

        
    
def get_projection_matrix(K, R=None, t=None):
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    return K @ np.hstack((R.T, R.T @ -t[:, None]))


def rotation_vector_to_matrix(rotation_vector):
    """
    Convert a rotation vector to a rotation matrix.
    
    The rotation vector is a vector whose magnitude is equal to the rotation angle (in radians)
    and whose direction is aligned with the axis of rotation.
    
    Args:
    rotation_vector (np.array): A 3D rotation vector.
    
    Returns:
    np.array: A 3x3 rotation matrix.
    """
    # Compute the angle (magnitude of the rotation vector)
    angle = np.linalg.norm(rotation_vector)

    # If the angle is very small, the rotation is approximately zero
    if np.isclose(angle, 0):
        return np.eye(3)

    # Normalize the rotation vector to get the rotation axis
    axis = rotation_vector / angle

    # Compute the components of the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x, y, z = axis

    # Using the Rodrigues' rotation formula
    R = np.array([
        [cos_angle + x*x*(1 - cos_angle),      x*y*(1 - cos_angle) - z*sin_angle, x*z*(1 - cos_angle) + y*sin_angle],
        [y*x*(1 - cos_angle) + z*sin_angle, cos_angle + y*y*(1 - cos_angle),      y*z*(1 - cos_angle) - x*sin_angle],
        [z*x*(1 - cos_angle) - y*sin_angle, z*y*(1 - cos_angle) + x*sin_angle, cos_angle + z*z*(1 - cos_angle)]
    ])

    return R
