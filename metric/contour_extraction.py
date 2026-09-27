"""
Extraccion de contornos a partir de mascaras de segmentacion.

La idea del asesor: dejar de comparar mascaras voxel-a-voxel y pasar a comparar
la SUPERFICIE (contorno 3D) del vaso. Aqui extraemos esa superficie como una nube
de puntos con su normal asociada, que es el descriptor que despues alimenta tanto
la metrica (matching humgaro) como la loss (Sinkhorn).

Para 2D el "contorno" es la curva de borde; para 3D es la superficie tubular.
Usamos marching cubes (skimage) que ya devuelve vertices + normales por vertice.
"""

from __future__ import annotations

import numpy as np
from skimage import measure


def extract_surface_points(
    mask: np.ndarray,
    level: float = 0.5,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    step_size: int = 1,
):
    """Extrae los puntos de superficie (contorno 3D) de una mascara binaria.

    Parameters
    ----------
    mask : np.ndarray
        Volumen 3D binario o de probabilidad (D, H, W).
    level : float
        Isovalor para marching cubes (0.5 para mascaras binarias).
    spacing : tuple
        Tamano de voxel (z, y, x). Importa para que las distancias esten en mm.
    step_size : int
        Submuestreo de marching cubes. >1 acelera y reduce la densidad de puntos.

    Returns
    -------
    points : (N, 3) float32   coordenadas de los puntos de superficie
    normals : (N, 3) float32  normal unitaria en cada punto
    """
    if mask.ndim != 3:
        raise ValueError(f"Se espera un volumen 3D, llego ndim={mask.ndim}")

    # marching_cubes necesita que el isovalor este dentro del rango de datos.
    mask = mask.astype(np.float32)
    if mask.max() <= level or mask.min() >= level:
        # mascara vacia o totalmente llena -> no hay superficie
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32))

    verts, faces, normals, _ = measure.marching_cubes(
        mask, level=level, spacing=spacing, step_size=step_size
    )
    return verts.astype(np.float32), normals.astype(np.float32)


def subsample(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    max_points: int = 1500,
    rng: np.random.Generator | None = None,
):
    """Submuestrea la nube de puntos a max_points.

    Si max_points es None, NO submuestrea (usa todos los puntos). En la metrica de
    evaluacion (offline, sin gradiente) usamos None + matching escalable por KD-tree
    (`match_contours_full`), de modo que cada nube tenga su numero real de puntos y
    FP/FN se desacoplen.
    """
    n = len(points)
    if max_points is None or n <= max_points:
        return points, normals
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
    sub_n = normals[idx] if normals is not None else None
    return points[idx], sub_n


def local_radius(mask, points, spacing=(1.0, 1.0, 1.0)):
    """Radio local del vaso en cada punto de superficie.

    El umbral natural del matching en vasos NO es un escalar fijo: un punto
    predicho esta "bien" si cae dentro del tubo del GT, o sea dentro del radio
    local. Aqui lo calculamos asi:
        1. distance transform de la mascara (distancia de cada voxel al fondo).
        2. esqueleto (centerline) del vaso.
        3. radio en cada voxel del centerline = valor del DT ahi (= distancia a
           la superficie = radio del vaso en ese punto).
        4. a cada punto de superficie le asignamos el radio del centerline mas
           cercano.

    Returns
    -------
    radii : (N,) radio local en mm para cada punto en `points`.
    """
    from scipy.ndimage import distance_transform_edt, maximum_filter
    from scipy.spatial import cKDTree

    m = mask.astype(bool)
    if m.sum() == 0 or len(points) == 0:
        return np.ones(len(points), np.float32)

    # distance transform (mm): el valor en cada voxel interior es su distancia a la
    # superficie. El radio local del vaso = DT en la "cresta" (linea central), donde
    # el DT es maximo. Aproximamos la cresta sin skeletonize (que cambia entre
    # versiones de skimage) tomando los maximos locales del DT, y a cada punto de
    # superficie le asignamos el radio de la cresta mas cercana.
    dt = distance_transform_edt(m, sampling=spacing)
    ridge = (dt == maximum_filter(dt, size=3)) & (dt > 0)   # crestas del DT
    rz, ry, rx = np.nonzero(ridge)
    if len(rz) == 0:                                        # respaldo: todo el interior
        rz, ry, rx = np.nonzero(m)

    ridge_coords = np.stack([rz * spacing[0], ry * spacing[1], rx * spacing[2]], 1)
    ridge_radius = np.clip(dt[rz, ry, rx].astype(np.float32), 0.5, None)

    tree = cKDTree(ridge_coords)
    _, idx = tree.query(points)
    return ridge_radius[idx]


def contour_descriptor(mask, max_points=1500, with_radius=False,
                       spacing=(1.0, 1.0, 1.0), **kwargs):
    """Atajo: mascara binaria -> (puntos, normales[, radio]) submuestreados.

    Este es el objeto que pasamos a la metrica y a la loss. Si with_radius=True
    devuelve tambien el radio local por punto, para usarlo como umbral del matching.
    """
    pts, nrm = extract_surface_points(mask, spacing=spacing, **kwargs)
    pts, nrm = subsample(pts, nrm, max_points=max_points)
    if not with_radius:
        return pts, nrm
    rad = local_radius(mask, pts, spacing=spacing)
    return pts, nrm, rad