"""
nnU-Net Validation Metrics Calculator
=====================================
Compara las predicciones de validación de nnU-Net contra las anotaciones ground truth
y calcula métricas por caso + promedios. Exporta a Excel.

Uso:
    python compute_metrics.py --labels /ruta/a/nnUNet_raw/DatasetXXX/labelsTr \
                              --preds  /ruta/a/nnUNet_results/.../fold_X/validation \
                              --output metricas_validacion.xlsx

    Si no quieres pasar argumentos, edita las rutas directamente abajo.
"""

import argparse
import os
import sys
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime


# =====================================================================
#  EDITA ESTAS RUTAS SI NO QUIERES USAR ARGUMENTOS DE LÍNEA DE COMANDO
# =====================================================================
DEFAULT_LABELS_DIR = ""   # ej: "/data/nnUNet_raw/Dataset001/labelsTr"
DEFAULT_PREDS_DIR  = ""   # ej: "/data/nnUNet_results/.../fold_0/validation"
DEFAULT_OUTPUT     = "metricas_validacion.xlsx"


def load_nifti(path):
    """Carga un NIfTI y retorna el array de datos como enteros."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    return (data > 0).astype(np.uint8), img.header.get_zooms()[:3]


def compute_metrics(pred, gt):
    """Calcula todas las métricas entre predicción y ground truth binarios."""
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    vol_sim = 1 - abs(fp - fn) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "vol_similarity": vol_sim,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def count_components(binary_vol):
    """Cuenta componentes conexas (6-conectividad) en un volumen binario."""
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    _, n_components = ndimage.label(binary_vol, structure=structure)
    return int(n_components)


def find_matching_cases(labels_dir, preds_dir):
    """
    Encuentra pares de archivos que coinciden entre labels y predicciones.
    nnU-Net validation genera archivos con el mismo nombre que los labels.
    """
    pred_files = {}
    for ext in ["*.nii.gz", "*.nii"]:
        for f in glob.glob(os.path.join(preds_dir, ext)):
            basename = os.path.basename(f)
            # Quitar extensión para matching
            name = basename.replace(".nii.gz", "").replace(".nii", "")
            pred_files[name] = f

    label_files = {}
    for ext in ["*.nii.gz", "*.nii"]:
        for f in glob.glob(os.path.join(labels_dir, ext)):
            basename = os.path.basename(f)
            name = basename.replace(".nii.gz", "").replace(".nii", "")
            label_files[name] = f

    # Encontrar intersección
    common = sorted(set(pred_files.keys()) & set(label_files.keys()))

    if not common:
        print(f"\n⚠️  No se encontraron casos coincidentes.")
        print(f"   Labels dir: {labels_dir} ({len(label_files)} archivos)")
        print(f"   Preds dir:  {preds_dir} ({len(pred_files)} archivos)")
        if label_files and pred_files:
            print(f"\n   Ejemplo labels: {list(label_files.keys())[:3]}")
            print(f"   Ejemplo preds:  {list(pred_files.keys())[:3]}")
        sys.exit(1)

    return [(name, label_files[name], pred_files[name]) for name in common]


def create_excel(results, output_path):
    """Crea un Excel profesional con las métricas por caso y promedios."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Métricas por Caso"

    # Estilos
    header_font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2563EB")
    data_font = Font(name="Arial", size=10)
    avg_font = Font(name="Arial", size=10, bold=True)
    avg_fill = PatternFill("solid", fgColor="E8F5E9")
    std_fill = PatternFill("solid", fgColor="FFF8E1")
    good_font = Font(name="Arial", size=10, color="059669")
    mid_font = Font(name="Arial", size=10, color="D97706")
    bad_font = Font(name="Arial", size=10, color="DC2626")
    border = Border(
        bottom=Side(style="thin", color="D8DDE8"),
        right=Side(style="thin", color="D8DDE8"),
    )
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")

    # Headers
    headers = [
        "Caso", "Dice (%)", "Precision (%)", "Recall (%)",
        "IoU (%)", "Vol. Similarity (%)",
        "TP (voxels)", "FP (voxels)", "FN (voxels)",
        "Comp. GT", "Comp. Pred"
    ]

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    # Datos
    pct_cols = [2, 3, 4, 5, 6]  # columnas con porcentajes
    for i, r in enumerate(results, 2):
        row_data = [
            r["case"],
            round(r["dice"] * 100, 2),
            round(r["precision"] * 100, 2),
            round(r["recall"] * 100, 2),
            round(r["iou"] * 100, 2),
            round(r["vol_similarity"] * 100, 2),
            r["tp"], r["fp"], r["fn"],
            r["comp_gt"], r["comp_pred"],
        ]
        for col, val in enumerate(row_data, 1):
            cell = ws.cell(row=i, column=col, value=val)
            cell.font = data_font
            cell.border = border
            cell.alignment = center if col > 1 else left

            # Color condicional para métricas porcentuales
            if col in pct_cols and isinstance(val, (int, float)):
                if val >= 80:
                    cell.font = good_font
                elif val >= 60:
                    cell.font = mid_font
                else:
                    cell.font = bad_font

    n = len(results)
    last_data_row = n + 1

    # Fila de PROMEDIO
    avg_row = last_data_row + 1
    ws.cell(row=avg_row, column=1, value="PROMEDIO").font = avg_font
    ws.cell(row=avg_row, column=1).fill = avg_fill
    ws.cell(row=avg_row, column=1).border = border
    ws.cell(row=avg_row, column=1).alignment = left

    for col in range(2, len(headers) + 1):
        col_letter = get_column_letter(col)
        cell = ws.cell(
            row=avg_row, column=col,
            value=f"=AVERAGE({col_letter}2:{col_letter}{last_data_row})"
        )
        cell.font = avg_font
        cell.fill = avg_fill
        cell.alignment = center
        cell.border = border
        if col in pct_cols:
            cell.number_format = "0.00"

    # Fila de STD
    std_row = avg_row + 1
    ws.cell(row=std_row, column=1, value="STD").font = avg_font
    ws.cell(row=std_row, column=1).fill = std_fill
    ws.cell(row=std_row, column=1).border = border
    ws.cell(row=std_row, column=1).alignment = left

    for col in range(2, len(headers) + 1):
        col_letter = get_column_letter(col)
        cell = ws.cell(
            row=std_row, column=col,
            value=f"=STDEV({col_letter}2:{col_letter}{last_data_row})"
        )
        cell.font = avg_font
        cell.fill = std_fill
        cell.alignment = center
        cell.border = border
        if col in pct_cols:
            cell.number_format = "0.00"

    # Fila MIN
    min_row = std_row + 1
    ws.cell(row=min_row, column=1, value="MIN").font = avg_font
    ws.cell(row=min_row, column=1).border = border
    ws.cell(row=min_row, column=1).alignment = left

    for col in range(2, len(headers) + 1):
        col_letter = get_column_letter(col)
        cell = ws.cell(
            row=min_row, column=col,
            value=f"=MIN({col_letter}2:{col_letter}{last_data_row})"
        )
        cell.font = data_font
        cell.alignment = center
        cell.border = border

    # Fila MAX
    max_row = min_row + 1
    ws.cell(row=max_row, column=1, value="MAX").font = avg_font
    ws.cell(row=max_row, column=1).border = border
    ws.cell(row=max_row, column=1).alignment = left

    for col in range(2, len(headers) + 1):
        col_letter = get_column_letter(col)
        cell = ws.cell(
            row=max_row, column=col,
            value=f"=MAX({col_letter}2:{col_letter}{last_data_row})"
        )
        cell.font = data_font
        cell.alignment = center
        cell.border = border

    # Anchos de columna
    widths = [28, 12, 14, 12, 10, 16, 14, 14, 14, 12, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # Freeze panes
    ws.freeze_panes = "B2"

    # ---- Sheet 2: Resumen del experimento ----
    ws2 = wb.create_sheet("Resumen")
    ws2.column_dimensions["A"].width = 25
    ws2.column_dimensions["B"].width = 30

    info = [
        ("Fecha de ejecución", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Casos evaluados", n),
        ("", ""),
        ("Dice promedio (%)", f"=AVERAGE('Métricas por Caso'!B2:B{last_data_row})"),
        ("Dice std (%)", f"=STDEV('Métricas por Caso'!B2:B{last_data_row})"),
        ("Precision promedio (%)", f"=AVERAGE('Métricas por Caso'!C2:C{last_data_row})"),
        ("Recall promedio (%)", f"=AVERAGE('Métricas por Caso'!D2:D{last_data_row})"),
        ("IoU promedio (%)", f"=AVERAGE('Métricas por Caso'!E2:E{last_data_row})"),
        ("Vol. Similarity promedio (%)", f"=AVERAGE('Métricas por Caso'!F2:F{last_data_row})"),
    ]

    for i, (label, value) in enumerate(info, 1):
        ws2.cell(row=i, column=1, value=label).font = Font(name="Arial", size=10, bold=True)
        cell = ws2.cell(row=i, column=2, value=value)
        cell.font = Font(name="Arial", size=10)
        cell.alignment = Alignment(horizontal="left")

    wb.save(output_path)
    print(f"\n✅ Excel guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calcula métricas de segmentación nnU-Net: predicción vs ground truth"
    )
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS_DIR,
                        help="Carpeta con los labels ground truth (.nii.gz)")
    parser.add_argument("--preds", type=str, default=DEFAULT_PREDS_DIR,
                        help="Carpeta con las predicciones de validación (.nii.gz)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Ruta del archivo Excel de salida")
    args = parser.parse_args()

    labels_dir = args.labels
    preds_dir = args.preds
    output_path = args.output

    if not labels_dir or not preds_dir:
        print("Error: Debes especificar --labels y --preds")
        print("  Ejemplo:")
        print("    python compute_metrics.py \\")
        print("      --labels /data/nnUNet_raw/Dataset001/labelsTr \\")
        print("      --preds  /data/nnUNet_results/.../fold_0/validation")
        sys.exit(1)

    if not os.path.isdir(labels_dir):
        print(f"Error: Carpeta de labels no encontrada: {labels_dir}")
        sys.exit(1)
    if not os.path.isdir(preds_dir):
        print(f"Error: Carpeta de predicciones no encontrada: {preds_dir}")
        sys.exit(1)

    # Encontrar pares
    cases = find_matching_cases(labels_dir, preds_dir)
    print(f"\n🔍 Encontrados {len(cases)} casos coincidentes entre labels y predicciones\n")
    print(f"{'Caso':<30} {'Dice':>8} {'Prec':>8} {'Rec':>8} {'IoU':>8} {'C_GT':>6} {'C_Pred':>6}")
    print("─" * 80)

    results = []

    for i, (name, label_path, pred_path) in enumerate(cases):
        gt_bin, _ = load_nifti(label_path)
        pred_bin, _ = load_nifti(pred_path)

        if gt_bin.shape != pred_bin.shape:
            print(f"  ⚠️  {name}: dimensiones no coinciden "
                  f"(GT={gt_bin.shape}, Pred={pred_bin.shape}), saltando...")
            continue

        m = compute_metrics(pred_bin, gt_bin)
        comp_gt = count_components(gt_bin)
        comp_pred = count_components(pred_bin)

        result = {
            "case": name,
            **m,
            "comp_gt": comp_gt,
            "comp_pred": comp_pred,
        }
        results.append(result)

        print(f"  {name:<28} {m['dice']*100:>7.2f}% {m['precision']*100:>7.2f}% "
              f"{m['recall']*100:>7.2f}% {m['iou']*100:>7.2f}% {comp_gt:>5} {comp_pred:>5}")

    if not results:
        print("\n❌ No se procesaron casos. Verifica las carpetas.")
        sys.exit(1)

    # Resumen en consola
    dices = [r["dice"] for r in results]
    print("─" * 80)
    print(f"  {'PROMEDIO':<28} {np.mean(dices)*100:>7.2f}%")
    print(f"  {'STD':<28} {np.std(dices)*100:>7.2f}%")
    print(f"  {'MIN':<28} {np.min(dices)*100:>7.2f}%")
    print(f"  {'MAX':<28} {np.max(dices)*100:>7.2f}%")

    # Exportar
    create_excel(results, output_path)
    print(f"   {len(results)} casos × 10 métricas + promedio/std/min/max\n")


if __name__ == "__main__":
    main()
