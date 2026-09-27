"""
==============================================================================
Betti Matching Error (τ_err) — Script dedicado
==============================================================================
Calcula exclusivamente el Betti Matching Error (Stucki et al., 2024) sobre
las predicciones de validación de los 5 experimentos del paper clCE.

Referencia:
  Stucki, Bürgin, Paetzold, Bauer. "Efficient Betti Matching Enables
  Topology-Aware 3D Segmentation via Persistent Homology." 2024.
  https://arxiv.org/abs/2407.04683
  https://github.com/nstucki/Betti-Matching-3D

Métricas:
  τ_err   = τ₀_err + τ₁_err (total)
  τ₀_err  = features dim-0 (componentes conexas) sin match espacial
  τ₁_err  = features dim-1 (loops/ciclos) sin match espacial

INSTALACIÓN:
  cd /path/to/project
  git clone https://github.com/nstucki/Betti-Matching-3D.git
  cd Betti-Matching-3D
  mkdir build && cd build
  cmake ..
  make
  export PYTHONPATH=$(pwd):$PYTHONPATH

Uso:
  python compute_betti_matching.py
  python compute_betti_matching.py --experiment "L_Dice"
  python compute_betti_matching.py --pred /path/to/pred.nii.gz --gt /path/to/gt.nii.gz

Requiere:
  - betti_matching (módulo C++ compilado de Betti-Matching-3D)
  - numpy, nibabel, pandas
==============================================================================
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# IMPORTAR BETTI MATCHING
# ============================================================================

try:
    import betti_matching
    print("[OK] betti_matching cargado")
except ImportError:
    print("[ERROR] No se pudo importar betti_matching.")
    print()
    print("  Instrucciones de instalación:")
    print("  1. git clone https://github.com/nstucki/Betti-Matching-3D.git")
    print("  2. cd Betti-Matching-3D && mkdir build && cd build")
    print("  3. cmake .. && make")
    print("  4. export PYTHONPATH=/ruta/a/Betti-Matching-3D/build:$PYTHONPATH")
    print()
    sys.exit(1)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_DIR = "/path/to/project"
GT_DIR = os.path.join(BASE_DIR, "nnUnet_raw", "Dataset002_Imagecas", "labelsTr")
RESULTS_DIR = os.path.join(BASE_DIR, "nnUNet_results", "Dataset002_Imagecas")

EXPERIMENTS = {
    "L_Dice": "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "L_Dice+0.5*L_clDice": "nnUNetTrainerDiceclDiceLoss__nnUNetPlans__3d_fullres",
    "L_Dice+L_clCE": "nnUNetTrainerDiceclCELoss__nnUNetPlans__3d_fullres",
    "L_CE+0.5*L_clDice": "nnUNetTrainerCEclDiceLoss__nnUNetPlans__3d_fullres",
    "L_CE+L_clCE": "nnUNetTrainerCEclCELoss__nnUNetPlans__3d_fullres",
}

FOLD = "fold_0"


# ============================================================================
# CÁLCULO DE BETTI MATCHING
# ============================================================================

def compute_betti_matching(pred_mask, gt_mask, filtration='superlevel',
                           construction='V', relative=False):
    """
    Calcula el Betti Matching Error entre una predicción y un ground truth.

    Args:
        pred_mask: np.ndarray binario (predicción)
        gt_mask:   np.ndarray binario (ground truth)
        filtration: 'superlevel' (foreground) o 'sublevel' (background)
        construction: 'V' (V-construction) o 'T' (T-construction)
        relative: bool, usar relative Betti matching

    Returns:
        dict con tau_err, tau0_err, tau1_err y metadata
    """
    pred_f = pred_mask.astype(np.float32)
    gt_f = gt_mask.astype(np.float32)

    t0 = time.time()

    BM = betti_matching.BettiMatching(
        pred_f, gt_f,
        relative=relative,
        comparison='union',
        filtration=filtration,
        construction=construction
    )

    # τ_err por dimensión
    tau0 = BM.loss(dimensions=[0])
    tau1 = BM.loss(dimensions=[1])
    tau_total = BM.loss(dimensions=[0, 1])

    # β_err para comparación directa
    beta0 = BM.Betti_number_error(threshold=0.5, dimensions=[0])
    beta1 = BM.Betti_number_error(threshold=0.5, dimensions=[1])
    beta_total = BM.Betti_number_error(threshold=0.5, dimensions=[0, 1])

    elapsed = time.time() - t0

    return {
        'tau_err': tau_total,
        'tau0_err': tau0,
        'tau1_err': tau1,
        'beta_err': beta_total,
        'beta0_err': beta0,
        'beta1_err': beta1,
        'compute_time_s': elapsed,
    }


# ============================================================================
# MODO: PAR INDIVIDUAL (--pred y --gt)
# ============================================================================

def run_single_pair(pred_path, gt_path):
    """Calcula Betti matching para un solo par pred/gt."""
    print(f"\n  Pred: {pred_path}")
    print(f"  GT:   {gt_path}\n")

    pred = nib.load(pred_path).get_fdata().astype(np.float32)
    gt = nib.load(gt_path).get_fdata().astype(np.float32)

    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)

    print(f"  Shape: {pred_bin.shape}")
    print(f"  Pred voxels: {pred_bin.sum():,}")
    print(f"  GT voxels:   {gt_bin.sum():,}")
    print(f"\n  Calculando Betti matching...", flush=True)

    result = compute_betti_matching(pred_bin, gt_bin)

    print(f"\n  {'='*50}")
    print(f"  RESULTADOS")
    print(f"  {'='*50}")
    print(f"  Betti Matching Error (τ):")
    print(f"    τ_err total : {result['tau_err']:.0f}")
    print(f"    τ₀_err      : {result['tau0_err']:.0f}  (componentes conexas)")
    print(f"    τ₁_err      : {result['tau1_err']:.0f}  (loops/ciclos)")
    print(f"  Betti Number Error (β) [comparación]:")
    print(f"    β_err total : {result['beta_err']:.0f}")
    print(f"    β₀_err      : {result['beta0_err']:.0f}")
    print(f"    β₁_err      : {result['beta1_err']:.0f}")
    print(f"  Tiempo: {result['compute_time_s']:.2f}s")
    print(f"  {'='*50}\n")

    return result


# ============================================================================
# MODO: BATCH (todos los experimentos)
# ============================================================================

def find_validation_predictions(experiment_dir):
    for subdir in ["validation", "validation_raw", "validation_raw_postprocessed"]:
        path = os.path.join(experiment_dir, FOLD, subdir)
        if os.path.isdir(path):
            files = sorted([f for f in os.listdir(path)
                            if f.endswith('.nii.gz') and not f.startswith('.')])
            if files:
                return path, files
    return None, []


def find_gt(case_filename, gt_dir):
    direct = os.path.join(gt_dir, case_filename)
    if os.path.exists(direct):
        return direct
    case_id = case_filename.replace('.nii.gz', '')
    for f in os.listdir(gt_dir):
        if f.endswith('.nii.gz'):
            fid = f.replace('.nii.gz', '')
            if case_id == fid or case_id in fid or fid in case_id:
                return os.path.join(gt_dir, f)
    return None


def run_batch(experiments, gt_dir, output_dir):
    """Corre Betti matching sobre todos los experimentos."""
    all_results = []

    for exp_name, exp_folder in experiments.items():
        exp_dir = os.path.join(RESULTS_DIR, exp_folder)

        print(f"\n{'─'*70}")
        print(f"  {exp_name}")
        print(f"  {exp_folder}")
        print(f"{'─'*70}")

        if not os.path.isdir(exp_dir):
            print(f"  [X] No encontrado: {exp_dir}")
            continue

        val_dir, val_files = find_validation_predictions(exp_dir)
        if not val_dir:
            print(f"  [X] Sin predicciones de validación")
            fold_dir = os.path.join(exp_dir, FOLD)
            if os.path.isdir(fold_dir):
                print(f"    Contenido: {os.listdir(fold_dir)}")
            continue

        print(f"  [OK] {val_dir} ({len(val_files)} casos)")

        for i, pred_file in enumerate(val_files):
            case_id = pred_file.replace('.nii.gz', '')
            print(f"    [{i+1}/{len(val_files)}] {case_id}...", end=' ', flush=True)

            gt_path = find_gt(pred_file, gt_dir)
            if not gt_path:
                print("GT no encontrado")
                continue

            try:
                pred = nib.load(os.path.join(val_dir, pred_file)).get_fdata()
                gt = nib.load(gt_path).get_fdata()

                if pred.shape != gt.shape:
                    print(f"Shape mismatch: {pred.shape} vs {gt.shape}")
                    continue

                pred_bin = (pred > 0).astype(bool)
                gt_bin = (gt > 0).astype(bool)

                result = compute_betti_matching(pred_bin, gt_bin)
                result['case_id'] = case_id
                result['experiment'] = exp_name
                all_results.append(result)

                print(f"tau={result['tau_err']:.0f} "
                      f"(t0={result['tau0_err']:.0f}, t1={result['tau1_err']:.0f}) "
                      f"beta={result['beta_err']:.0f} "
                      f"[{result['compute_time_s']:.1f}s]")

            except Exception as e:
                print(f"Error: {e}")

        # Guardar CSV por experimento
        if any(r['experiment'] == exp_name for r in all_results):
            exp_df = pd.DataFrame([r for r in all_results if r['experiment'] == exp_name])
            csv_path = os.path.join(output_dir,
                                    f"betti_matching_{exp_folder.split('__')[0]}.csv")
            exp_df.to_csv(csv_path, index=False)
            print(f"\n  Guardado: {csv_path}")

    return all_results


def print_summary(df):
    """Tabla resumen comparando τ_err vs β_err por experimento."""
    metrics = ['tau_err', 'tau0_err', 'tau1_err', 'beta_err', 'beta0_err', 'beta1_err']

    rows = []
    for exp in EXPERIMENTS.keys():
        edf = df[df['experiment'] == exp]
        if len(edf) == 0:
            continue
        row = {'Experiment': exp, 'N': len(edf)}
        for m in metrics:
            vals = edf[m].dropna()
            if len(vals) > 0:
                row[m] = f"{vals.mean():.1f} +/- {vals.std():.1f}"
            else:
                row[m] = "N/A"
        avg_time = edf['compute_time_s'].mean()
        row['avg_time'] = f"{avg_time:.1f}s"
        rows.append(row)

    sdf = pd.DataFrame(rows)

    print(f"\n{'='*130}")
    print(f"  BETTI MATCHING ERROR vs BETTI NUMBER ERROR")
    print(f"  Dataset002_ImageCAS | fold_0")
    print(f"{'='*130}")
    try:
        from tabulate import tabulate
        print(tabulate(sdf, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        print(sdf.to_string(index=False))
    print(f"{'='*130}")
    print("  tau_err >= beta_err siempre (tau es un refinamiento de beta)")
    print("  tau_err = beta_err cuando todas las features coinciden espacialmente")
    print("  tau_err > beta_err cuando hay features en la posición incorrecta")
    print(f"{'='*130}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Betti Matching Error (tau_err) para segmentación vascular'
    )

    # Modo par individual
    parser.add_argument('--pred', type=str, default=None,
                        help='Path a predicción .nii.gz (modo par individual)')
    parser.add_argument('--gt', type=str, default=None,
                        help='Path a ground truth .nii.gz (modo par individual)')

    # Modo batch
    parser.add_argument('--experiment', type=str, default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help='Solo un experimento')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(BASE_DIR, "topology_results"),
                        help='Directorio de salida')
    parser.add_argument('--gt_dir', type=str, default=GT_DIR)

    args = parser.parse_args()

    # ---- MODO PAR INDIVIDUAL ----
    if args.pred and args.gt:
        print(f"\n{'='*60}")
        print(f"  BETTI MATCHING - Par individual")
        print(f"{'='*60}")
        run_single_pair(args.pred, args.gt)
        return

    if args.pred or args.gt:
        print("ERROR: --pred y --gt deben usarse juntos")
        sys.exit(1)

    # ---- MODO BATCH ----
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BETTI MATCHING ERROR - Batch")
    print(f"  Dataset002_ImageCAS | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"  GT:     {args.gt_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(args.gt_dir):
        print(f"\nERROR: GT dir no encontrado: {args.gt_dir}")
        sys.exit(1)

    experiments = ({args.experiment: EXPERIMENTS[args.experiment]}
                   if args.experiment else EXPERIMENTS)

    all_results = run_batch(experiments, args.gt_dir, args.output_dir)

    if not all_results:
        print("\n[WARN] No se generaron resultados.")
        sys.exit(1)

    # Guardar combinado
    df = pd.DataFrame(all_results)
    combined = os.path.join(args.output_dir, "betti_matching_all_imagecas.csv")
    df.to_csv(combined, index=False)

    # Resumen
    print_summary(df)

    print(f"  Resultados: {combined}")
    print(f"\nProceso completado!\n")


if __name__ == '__main__':
    main()
