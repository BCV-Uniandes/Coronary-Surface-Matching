"""
==============================================================================
Cálculo de Métricas de Topología - ImageCAS (Dataset002)
==============================================================================
Calcula métricas de topología sobre las predicciones de validación (fold_0)
de los 5 experimentos del paper "Centerline-Cross Entropy Loss".

Métricas:
  - DSC (Dice Similarity Coefficient)
  - clDice (Centerline Dice)
  - β₀ error (error en componentes conexas)
  - β₁ error (error en loops/ciclos)
  - Betti error total
  - Euler characteristic error
  - Número de componentes conexas (pred y GT)

Uso:
  python compute_topology_imagecas.py

  # Si quieres solo un experimento:
  python compute_topology_imagecas.py --experiment L_Dice

  # Si quieres guardar en otra ruta:
  python compute_topology_imagecas.py --output_dir ./mis_resultados/

Requiere:
  pip install numpy scipy scikit-image nibabel pandas tabulate
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import nibabel as nib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN DE PATHS  (adaptada a tu estructura)
# ============================================================================

BASE_DIR = "/path/to/project"

# Ground truth labels (training set, del cual salen los validation cases)
GT_DIR = os.path.join(BASE_DIR, "nnUnet_raw", "Dataset002_Imagecas", "labelsTr")

# Directorio base de resultados
RESULTS_DIR = os.path.join(BASE_DIR, "nnUNet_results", "Dataset002_Imagecas")

# Mapeo: nombre legible → carpeta del trainer en nnUNet
EXPERIMENTS = {
    "L_Dice": "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "L_Dice+0.5*L_clDice": "nnUNetTrainerDiceclDiceLoss__nnUNetPlans__3d_fullres",
    "L_Dice+L_clCE": "nnUNetTrainerDiceclCELoss__nnUNetPlans__3d_fullres",
    "L_CE+0.5*L_clDice": "nnUNetTrainerCEclDiceLoss__nnUNetPlans__3d_fullres",
    "L_CE+L_clCE": "nnUNetTrainerCEclCELoss__nnUNetPlans__3d_fullres",
}

FOLD = "fold_0"


# ============================================================================
# MÉTRICAS DE TOPOLOGÍA
# ============================================================================

def compute_euler_characteristic_3d(mask):
    """
    Característica de Euler para imagen binaria 3D via cubical complexes.
    χ = V - E + F - C
    """
    m = mask.astype(bool)
    V = int(m.sum())
    if V == 0:
        return 0

    # Aristas (pares adyacentes en 3 ejes)
    E = (int(np.logical_and(m[:-1, :, :], m[1:, :, :]).sum()) +
         int(np.logical_and(m[:, :-1, :], m[:, 1:, :]).sum()) +
         int(np.logical_and(m[:, :, :-1], m[:, :, 1:]).sum()))

    # Caras (cuartetos coplanares en 3 planos)
    F = (int(np.logical_and(
            np.logical_and(m[:-1, :-1, :], m[1:, :-1, :]),
            np.logical_and(m[:-1, 1:, :], m[1:, 1:, :])).sum()) +
         int(np.logical_and(
            np.logical_and(m[:-1, :, :-1], m[1:, :, :-1]),
            np.logical_and(m[:-1, :, 1:], m[1:, :, 1:])).sum()) +
         int(np.logical_and(
            np.logical_and(m[:, :-1, :-1], m[:, 1:, :-1]),
            np.logical_and(m[:, :-1, 1:], m[:, 1:, 1:])).sum()))

    # Cubos (octetos)
    C = int(np.logical_and(
        np.logical_and(
            np.logical_and(m[:-1, :-1, :-1], m[1:, :-1, :-1]),
            np.logical_and(m[:-1, 1:, :-1], m[1:, 1:, :-1])),
        np.logical_and(
            np.logical_and(m[:-1, :-1, 1:], m[1:, :-1, 1:]),
            np.logical_and(m[:-1, 1:, 1:], m[1:, 1:, 1:]))).sum())

    return V - E + F - C


def compute_betti_numbers_3d(mask):
    """
    β₀ = componentes conexas, β₁ = loops, β₂ ≈ 0 para vasos.
    β₁ = β₀ - χ  (asumiendo β₂ = 0)
    """
    if mask.sum() == 0:
        return 0, 0

    _, beta_0 = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    euler = compute_euler_characteristic_3d(mask)
    beta_1 = max(0, beta_0 - euler)

    return int(beta_0), int(beta_1)


def cl_score(volume, skeleton):
    """Fracción del esqueleto cubierta por el volumen."""
    if skeleton.sum() == 0:
        return 1.0 if volume.sum() == 0 else 0.0
    return float(np.logical_and(skeleton, volume).sum()) / float(skeleton.sum())


def compute_cldice(pred, gt):
    """
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    Tprec = |S(GT) ∩ pred| / |S(GT)|
    Tsens = |S(pred) ∩ GT| / |S(pred)|
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0

    skel_pred = skeletonize_3d(pred.astype(np.uint8)).astype(bool)
    skel_gt = skeletonize_3d(gt.astype(np.uint8)).astype(bool)

    tprec = cl_score(pred, skel_gt)
    tsens = cl_score(gt, skel_pred)

    if tprec + tsens == 0:
        return 0.0, tprec, tsens
    cldice = 2.0 * tprec * tsens / (tprec + tsens)
    return cldice, tprec, tsens


def dice_coefficient(pred, gt):
    """DSC estándar."""
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_all_metrics(pred_mask, gt_mask):
    """Calcula todas las métricas para un par predicción/GT."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    # DSC
    dsc = dice_coefficient(pred, gt)

    # clDice
    cldice, tprec, tsens = compute_cldice(pred, gt)

    # Betti numbers
    b0_pred, b1_pred = compute_betti_numbers_3d(pred)
    b0_gt, b1_gt = compute_betti_numbers_3d(gt)

    # Euler
    euler_pred = compute_euler_characteristic_3d(pred)
    euler_gt = compute_euler_characteristic_3d(gt)

    # Componentes conexas
    _, n_comp_pred = ndimage.label(pred, structure=ndimage.generate_binary_structure(3, 3))
    _, n_comp_gt = ndimage.label(gt, structure=ndimage.generate_binary_structure(3, 3))

    return {
        'DSC': dsc,
        'clDice': cldice,
        'Tprec': tprec,
        'Tsens': tsens,
        'B0_pred': b0_pred,
        'B0_gt': b0_gt,
        'B0_error': abs(b0_pred - b0_gt),
        'B1_pred': b1_pred,
        'B1_gt': b1_gt,
        'B1_error': abs(b1_pred - b1_gt),
        'Betti_error': abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt),
        'Euler_pred': euler_pred,
        'Euler_gt': euler_gt,
        'Euler_error': abs(euler_pred - euler_gt),
        'N_comp_pred': n_comp_pred,
        'N_comp_gt': n_comp_gt,
    }


# ============================================================================
# PIPELINE
# ============================================================================

def find_validation_predictions(experiment_dir):
    """
    Busca predicciones de validación en la estructura nnUNet.
    Posibles ubicaciones:
      fold_0/validation/
      fold_0/validation_raw/
    """
    candidates = [
        os.path.join(experiment_dir, FOLD, "validation"),
        os.path.join(experiment_dir, FOLD, "validation_raw"),
        # nnUNet v1 legacy
        os.path.join(experiment_dir, FOLD, "validation_raw_postprocessed"),
    ]

    for path in candidates:
        if os.path.isdir(path):
            nii_files = [f for f in os.listdir(path)
                         if f.endswith('.nii.gz') and not f.startswith('.')]
            if nii_files:
                return path, nii_files

    return None, []


def find_gt_for_case(case_filename, gt_dir):
    """
    Dado un filename de predicción, encuentra el GT correspondiente.
    nnUNet predictions suelen tener el mismo nombre que los labels.
    """
    # Nombre directo
    gt_path = os.path.join(gt_dir, case_filename)
    if os.path.exists(gt_path):
        return gt_path

    # A veces el label tiene _0000 removido o diferente sufijo
    case_id = case_filename.replace('.nii.gz', '')

    for f in os.listdir(gt_dir):
        if not f.endswith('.nii.gz'):
            continue
        gt_id = f.replace('.nii.gz', '')
        if case_id == gt_id or case_id in gt_id or gt_id in case_id:
            return os.path.join(gt_dir, f)

    return None


def process_experiment(exp_name, exp_folder, gt_dir):
    """Procesa un experimento completo."""
    exp_dir = os.path.join(RESULTS_DIR, exp_folder)

    if not os.path.isdir(exp_dir):
        print(f"  ✗ Directorio no encontrado: {exp_dir}")
        return []

    val_dir, val_files = find_validation_predictions(exp_dir)
    if val_dir is None:
        print(f"  ✗ No se encontraron predicciones de validación en {exp_dir}")
        # Listar contenido para debug
        fold_dir = os.path.join(exp_dir, FOLD)
        if os.path.isdir(fold_dir):
            print(f"    Contenido de {FOLD}/: {os.listdir(fold_dir)}")
        return []

    print(f"  ✓ Predicciones encontradas: {val_dir} ({len(val_files)} archivos)")

    results = []
    for i, pred_file in enumerate(sorted(val_files)):
        case_id = pred_file.replace('.nii.gz', '')
        print(f"    [{i+1}/{len(val_files)}] {case_id}...", end=' ', flush=True)

        pred_path = os.path.join(val_dir, pred_file)
        gt_path = find_gt_for_case(pred_file, gt_dir)

        if gt_path is None:
            print(f"⚠ GT no encontrado, saltando")
            continue

        try:
            pred_img = nib.load(pred_path)
            gt_img = nib.load(gt_path)

            pred_data = pred_img.get_fdata().astype(np.float32)
            gt_data = gt_img.get_fdata().astype(np.float32)

            if pred_data.shape != gt_data.shape:
                print(f"⚠ Shapes no coinciden: pred={pred_data.shape} gt={gt_data.shape}")
                continue

            metrics = compute_all_metrics(pred_data, gt_data)
            metrics['case_id'] = case_id
            metrics['experiment'] = exp_name
            results.append(metrics)

            print(f"DSC={metrics['DSC']:.3f}  clDice={metrics['clDice']:.3f}  "
                  f"β₀err={metrics['B0_error']}  β₁err={metrics['B1_error']}")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    return results


def print_summary_table(df):
    """Imprime tabla resumen formateada."""
    metrics_to_show = ['DSC', 'clDice', 'B0_error', 'B1_error',
                       'Betti_error', 'Euler_error']

    summary_rows = []
    for exp in EXPERIMENTS.keys():
        exp_df = df[df['experiment'] == exp]
        if len(exp_df) == 0:
            continue
        row = {'Experiment': exp, 'N': len(exp_df)}
        for m in metrics_to_show:
            mean = exp_df[m].mean()
            std = exp_df[m].std()
            row[f'{m}'] = f"{mean:.4f}±{std:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 100)
    print("  TABLA RESUMEN - MÉTRICAS DE TOPOLOGÍA - Dataset002_ImageCAS (fold_0)")
    print("=" * 100)

    try:
        from tabulate import tabulate
        print(tabulate(summary_df, headers='keys', tablefmt='grid',
                       showindex=False))
    except ImportError:
        print(summary_df.to_string(index=False))

    print("=" * 100)
    print("  ↑ = mayor es mejor (DSC, clDice)")
    print("  ↓ = menor es mejor (B0_error, B1_error, Betti_error, Euler_error)")
    print("=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Métricas de topología para segmentación vascular - ImageCAS'
    )
    parser.add_argument('--experiment', type=str, default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help='Ejecutar solo un experimento específico')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(BASE_DIR, "topology_results"),
                        help='Directorio para guardar resultados CSV')
    parser.add_argument('--gt_dir', type=str, default=GT_DIR,
                        help='Override del directorio de ground truth')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MÉTRICAS DE TOPOLOGÍA - Dataset002_ImageCAS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"  GT dir:      {args.gt_dir}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Fold:        {FOLD}")
    print(f"  Output:      {args.output_dir}")
    print(f"{'='*70}\n")

    # Verificar que el GT dir existe
    if not os.path.isdir(args.gt_dir):
        print(f"ERROR: No se encontró el directorio de GT: {args.gt_dir}")
        print("Verifica el path y vuelve a ejecutar.")
        sys.exit(1)

    gt_files = [f for f in os.listdir(args.gt_dir) if f.endswith('.nii.gz')]
    print(f"Ground truth: {len(gt_files)} archivos en {args.gt_dir}\n")

    # Seleccionar experimentos
    if args.experiment:
        experiments = {args.experiment: EXPERIMENTS[args.experiment]}
    else:
        experiments = EXPERIMENTS

    all_results = []

    for exp_name, exp_folder in experiments.items():
        print(f"\n{'─'*70}")
        print(f"  Experimento: {exp_name}")
        print(f"  Trainer:     {exp_folder}")
        print(f"{'─'*70}")

        results = process_experiment(exp_name, exp_folder, args.gt_dir)
        all_results.extend(results)

        if results:
            exp_df = pd.DataFrame(results)
            csv_path = os.path.join(
                args.output_dir,
                f"topology_{exp_folder.split('__')[0]}.csv"
            )
            exp_df.to_csv(csv_path, index=False)
            print(f"\n  Guardado: {csv_path}")

    if not all_results:
        print("\n⚠ No se generaron resultados. Verifica:")
        print("  1. Que los modelos fueron entrenados (fold_0/)")
        print("  2. Que existen predicciones de validación (fold_0/validation/)")
        print("  3. Que los GT labels están en labelsTr/")
        print("\n  Si las predicciones no existen, genera las de validación con:")
        print("  nnUNetv2_predict -i <imagesTs> -o <output> -d 002 -c 3d_fullres "
              "-tr <TrainerName> -f 0")
        sys.exit(1)

    # Combinar todo
    df_all = pd.DataFrame(all_results)
    combined_path = os.path.join(args.output_dir, "all_topology_metrics_imagecas.csv")
    df_all.to_csv(combined_path, index=False)
    print(f"\n  Resultados combinados: {combined_path}")

    # Tabla resumen
    print_summary_table(df_all)

    # Guardar tabla resumen
    summary_path = os.path.join(args.output_dir, "summary_imagecas.csv")
    metrics_cols = ['DSC', 'clDice', 'B0_error', 'B1_error',
                    'Betti_error', 'Euler_error']
    summary = df_all.groupby('experiment')[metrics_cols].agg(['mean', 'std'])
    summary.to_csv(summary_path)
    print(f"\n  Resumen guardado: {summary_path}")
    print("\n¡Proceso completado!\n")


if __name__ == '__main__':
    main()
