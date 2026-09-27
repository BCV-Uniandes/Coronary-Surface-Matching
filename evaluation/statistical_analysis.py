"""
Análisis estadístico de métricas entre experimentos nnUNet
----------------------------------------------------------
1. Test de normalidad (Shapiro-Wilk) por métrica y experimento
2. Test de diferencia entre distribuciones:
   - Si TODAS las distribuciones son normales → ANOVA + Tukey HSD
   - Si alguna NO es normal → Kruskal-Wallis + Dunn (post-hoc)
3. Guarda resultados en Excel con una hoja por análisis

Uso:
    python statistical_analysis.py

Requiere:
    pip install pandas numpy scipy scikit_posthocs openpyxl
"""

import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
FILES = {
    "Baseline":      "/path/to/project/topology_nnUNetTrainer.xlsx",
    "Exp1_clDice":   "/path/to/project/Dice_ClDIce.xlsx",
    "Exp2_clCE":     "/path/to/project/CeclCELoss_Final.xlsx",
}

METRICS = ["DSC", "clDice", "B0_error", "B1_error", "Betti_error", "Euler_error"]

OUTPUT_EXCEL = "statistical_results.xlsx"
ALPHA = 0.05  # nivel de significancia

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

def load_data():
    """Carga los Excel y devuelve un DataFrame combinado."""
    dfs = []
    for exp_name, path in FILES.items():
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()  # limpiar espacios en nombres
            df["experiment"] = exp_name
            dfs.append(df)
            print(f"[OK] {exp_name}: {len(df)} casos | columnas: {list(df.columns)}")
        except FileNotFoundError:
            print(f"[ERROR] No se encontró: {path}")
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")

    if not dfs:
        raise RuntimeError("No se pudo cargar ningún archivo.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def check_metrics(df):
    """Verifica que las métricas requeridas existen en el DataFrame."""
    available = []
    missing = []
    for m in METRICS:
        if m in df.columns:
            available.append(m)
        else:
            missing.append(m)
    if missing:
        print(f"\n[AVISO] Métricas no encontradas en los archivos: {missing}")
        print(f"        Se analizarán solo: {available}")
    return available


# ─────────────────────────────────────────────
# TEST DE NORMALIDAD (Shapiro-Wilk)
# ─────────────────────────────────────────────

def test_normality(df, metrics):
    """
    Aplica Shapiro-Wilk a cada (experimento x métrica).
    Devuelve DataFrame con resultados y columna 'is_normal'.
    """
    rows = []
    experiments = df["experiment"].unique()

    for metric in metrics:
        for exp in experiments:
            data = df.loc[df["experiment"] == exp, metric].dropna().values
            if len(data) < 3:
                stat, p, normal = np.nan, np.nan, False
            else:
                stat, p = stats.shapiro(data)
                normal = p > ALPHA
            rows.append({
                "metric":      metric,
                "experiment":  exp,
                "n":           len(data),
                "shapiro_W":   round(stat, 4) if not np.isnan(stat) else np.nan,
                "shapiro_p":   round(p, 4)    if not np.isnan(p)    else np.nan,
                "is_normal":   normal,
                "interpretation": "Normal" if normal else "No normal",
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# TESTS DE DIFERENCIA
# ─────────────────────────────────────────────

def test_differences(df, metrics, normality_df):
    """
    Por cada métrica:
      - Si todos los grupos son normales → ANOVA de 1 vía + Tukey HSD
      - Si alguno no es normal         → Kruskal-Wallis + Dunn (Bonferroni)
    """
    global_rows = []   # resultado global (ANOVA o K-W)
    posthoc_rows = []  # comparaciones por pares
    experiments = df["experiment"].unique()

    for metric in metrics:
        # Decidir qué test usar según normalidad
        norm_for_metric = normality_df[normality_df["metric"] == metric]["is_normal"]
        all_normal = norm_for_metric.all()
        test_used = "ANOVA (paramétrico)" if all_normal else "Kruskal-Wallis (no paramétrico)"

        groups = [df.loc[df["experiment"] == exp, metric].dropna().values
                  for exp in experiments]

        # ── Test global
        if all_normal:
            stat, p = stats.f_oneway(*groups)
            test_stat_name = "F"
        else:
            stat, p = stats.kruskal(*groups)
            test_stat_name = "H"

        significant = p < ALPHA
        global_rows.append({
            "metric":        metric,
            "test":          test_used,
            "statistic":     round(stat, 4),
            "statistic_name": test_stat_name,
            "p_value":       round(p, 6),
            "significant":   significant,
            "interpretation": "Diferencia significativa ✓" if significant else "Sin diferencia significativa",
        })

        print(f"\n  [{metric}] {test_used}")
        print(f"    {test_stat_name}={stat:.4f}, p={p:.6f} → {'SIGNIFICATIVO' if significant else 'no significativo'}")

        # ── Post-hoc (siempre, para saber entre quiénes)
        if all_normal:
            # Tukey HSD requiere datos apilados con etiquetas
            all_vals = np.concatenate(groups)
            all_labels = np.concatenate([[exp] * len(g) for exp, g in zip(experiments, groups)])
            tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=ALPHA)
            tukey_df = pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=tukey._results_table.data[0]
            )
            for _, row in tukey_df.iterrows():
                posthoc_rows.append({
                    "metric":       metric,
                    "test":         "Tukey HSD",
                    "group1":       row["group1"],
                    "group2":       row["group2"],
                    "mean_diff":    round(float(row["meandiff"]), 6),
                    "p_adj":        round(float(row["p-adj"]), 6),
                    "significant":  str(row["reject"]) == "True",
                    "ci_lower":     round(float(row["lower"]), 6),
                    "ci_upper":     round(float(row["upper"]), 6),
                })
        else:
            # Dunn con corrección Bonferroni
            data_dict = {exp: g for exp, g in zip(experiments, groups)}
            dunn = sp.posthoc_dunn(
                [data_dict[e] for e in experiments],
                p_adjust="bonferroni"
            )
            dunn.index = experiments
            dunn.columns = experiments
            for i, e1 in enumerate(experiments):
                for j, e2 in enumerate(experiments):
                    if j <= i:
                        continue
                    p_adj = dunn.loc[e1, e2]
                    posthoc_rows.append({
                        "metric":      metric,
                        "test":        "Dunn (Bonferroni)",
                        "group1":      e1,
                        "group2":      e2,
                        "mean_diff":   round(
                            float(np.mean(data_dict[e1])) - float(np.mean(data_dict[e2])), 6
                        ),
                        "p_adj":       round(float(p_adj), 6),
                        "significant": p_adj < ALPHA,
                        "ci_lower":    np.nan,
                        "ci_upper":    np.nan,
                    })

    return pd.DataFrame(global_rows), pd.DataFrame(posthoc_rows)


# ─────────────────────────────────────────────
# ESTADÍSTICAS DESCRIPTIVAS
# ─────────────────────────────────────────────

def descriptive_stats(df, metrics):
    rows = []
    for metric in metrics:
        for exp in df["experiment"].unique():
            data = df.loc[df["experiment"] == exp, metric].dropna()
            rows.append({
                "metric":    metric,
                "experiment": exp,
                "n":         len(data),
                "mean":      round(data.mean(), 6),
                "std":       round(data.std(), 6),
                "median":    round(data.median(), 6),
                "min":       round(data.min(), 6),
                "max":       round(data.max(), 6),
                "Q1":        round(data.quantile(0.25), 6),
                "Q3":        round(data.quantile(0.75), 6),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────

def save_results(desc_df, norm_df, global_df, posthoc_df):
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        desc_df.to_excel(writer, sheet_name="Descriptivas", index=False)
        norm_df.to_excel(writer, sheet_name="Normalidad (Shapiro-Wilk)", index=False)
        global_df.to_excel(writer, sheet_name="Test Global (ANOVA o K-W)", index=False)
        posthoc_df.to_excel(writer, sheet_name="Post-hoc (Tukey o Dunn)", index=False)

    print(f"\n{'='*60}")
    print(f"Resultados guardados en: {OUTPUT_EXCEL}")
    print(f"  Hoja 1: Estadísticas descriptivas")
    print(f"  Hoja 2: Test de normalidad Shapiro-Wilk")
    print(f"  Hoja 3: Test global (ANOVA o Kruskal-Wallis)")
    print(f"  Hoja 4: Comparaciones post-hoc por pares")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ANÁLISIS ESTADÍSTICO — EXPERIMENTOS nnUNet")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1/4] Cargando archivos Excel...")
    df = load_data()
    metrics = check_metrics(df)

    if not metrics:
        print("[ERROR] Ninguna métrica encontrada. Revisa los nombres de columnas.")
        print(f"Columnas disponibles: {list(df.columns)}")
        return

    # 2. Estadísticas descriptivas
    print("\n[2/4] Calculando estadísticas descriptivas...")
    desc_df = descriptive_stats(df, metrics)

    # 3. Test de normalidad
    print("\n[3/4] Test de normalidad (Shapiro-Wilk, α=0.05)...")
    norm_df = test_normality(df, metrics)
    for metric in metrics:
        subset = norm_df[norm_df["metric"] == metric]
        normal_count = subset["is_normal"].sum()
        total = len(subset)
        decision = "→ usará ANOVA" if normal_count == total else "→ usará Kruskal-Wallis"
        print(f"  {metric}: {normal_count}/{total} grupos normales  {decision}")

    # 4. Tests de diferencia
    print("\n[4/4] Tests de diferencia entre distribuciones...")
    global_df, posthoc_df = test_differences(df, metrics, norm_df)

    # 5. Guardar
    save_results(desc_df, norm_df, global_df, posthoc_df)

    # 6. Resumen final en consola
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS SIGNIFICATIVOS")
    print("=" * 60)
    sig = global_df[global_df["significant"] == True]
    if len(sig) == 0:
        print("  Ninguna métrica mostró diferencias significativas entre experimentos.")
    else:
        for _, row in sig.iterrows():
            print(f"  ✓ {row['metric']:15s} | {row['test']:30s} | p={row['p_value']:.6f}")
            pairs = posthoc_df[
                (posthoc_df["metric"] == row["metric"]) &
                (posthoc_df["significant"] == True)
            ]
            for _, p in pairs.iterrows():
                diff_str = f"Δmean={p['mean_diff']:+.4f}" if not np.isnan(p["mean_diff"]) else ""
                print(f"      → {p['group1']} vs {p['group2']} | p_adj={p['p_adj']:.6f} | {diff_str}")


if __name__ == "__main__":
    main()
