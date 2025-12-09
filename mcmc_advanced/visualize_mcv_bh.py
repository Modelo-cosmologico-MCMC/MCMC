#!/usr/bin/env python3
"""
================================================================================
VISUALIZACION MCV-AGUJEROS NEGROS
================================================================================

Script de visualizacion para el modulo MCV-BH del MCMC.
Genera las figuras del documento de analisis.

Figuras generadas:
1. Parametros por rangos de masa
2. Perfiles radiales de MCV
3. Condiciones para diferentes S_local
4. Definicion de la burbuja entropica

Autor: Modelo MCMC
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import json
import os

# Importar el modulo principal
from mcv_bh_calibrated import (
    MCV_AgujerosNegros,
    crear_BH_canonico,
    analizar_por_categorias,
    EJEMPLOS_CANONICOS,
    SELLOS,
    XI_BUBBLE,
    XI_FREEZE,
    XI_COLLAPSE,
    S_EXT,
)

# Configuracion de matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def figura1_parametros_masa(save_path: str = None):
    """
    Figura 1: Parametros MCMC por rangos de masa de BH.

    Muestra como varian S_local, Delta_S, Xi y dilatacion temporal
    con la masa del agujero negro.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rango de masas
    log_M = np.linspace(-15, 12, 100)
    M_solar = 10**log_M

    # Calcular parametros para cada masa
    S_local = []
    Delta_S = []
    Xi_max = []
    dilatacion = []

    for M in M_solar:
        bh = MCV_AgujerosNegros(M)
        r_hor = bh.r_s * 1.01

        S_local.append(bh.S_local(r_hor))
        Delta_S.append(bh.Delta_S(r_hor))
        Xi_max.append(bh.Xi(r_hor))
        dilatacion.append(bh.factor_tiempo_relativo(r_hor))

    S_local = np.array(S_local)
    Delta_S = np.array(Delta_S)
    Xi_max = np.array(Xi_max)
    dilatacion = np.array(dilatacion)

    # Colores por categoria
    colors = []
    for M in M_solar:
        if M < 1e-5:
            colors.append('purple')
        elif M < 100:
            colors.append('blue')
        elif M < 1e5:
            colors.append('green')
        elif M < 1e10:
            colors.append('orange')
        else:
            colors.append('red')

    # Panel 1: S_local vs M
    ax1 = axes[0, 0]
    ax1.scatter(log_M, S_local, c=colors, s=10, alpha=0.7)
    ax1.axhline(y=S_EXT, color='gray', linestyle='--', label=f'S_ext = {S_EXT}')
    ax1.set_xlabel('log$_{10}$(M/M$_\odot$)')
    ax1.set_ylabel('S$_{local}$ (horizonte)')
    ax1.set_title('Entropia Local en el Horizonte')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Panel 2: Delta_S vs M
    ax2 = axes[0, 1]
    ax2.scatter(log_M, Delta_S, c=colors, s=10, alpha=0.7)
    ax2.set_xlabel('log$_{10}$(M/M$_\odot$)')
    ax2.set_ylabel('$\Delta$S = S$_{ext}$ - S$_{local}$')
    ax2.set_title('Friccion Entropica')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Xi vs M
    ax3 = axes[1, 0]
    ax3.scatter(log_M, np.log10(Xi_max + 1e-10), c=colors, s=10, alpha=0.7)
    ax3.axhline(y=np.log10(XI_FREEZE), color='blue', linestyle='--',
                label=f'Xi_freeze = {XI_FREEZE}')
    ax3.axhline(y=np.log10(XI_COLLAPSE), color='red', linestyle='--',
                label=f'Xi_collapse = {XI_COLLAPSE}')
    ax3.set_xlabel('log$_{10}$(M/M$_\odot$)')
    ax3.set_ylabel('log$_{10}$($\\Xi$)')
    ax3.set_title('Potencial Cronologico')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Dilatacion temporal vs M
    ax4 = axes[1, 1]
    ax4.scatter(log_M, np.log10(dilatacion + 1), c=colors, s=10, alpha=0.7)
    ax4.set_xlabel('log$_{10}$(M/M$_\odot$)')
    ax4.set_ylabel('log$_{10}$($\Delta$t/$\Delta$t$_0$)')
    ax4.set_title('Dilatacion Temporal')
    ax4.grid(True, alpha=0.3)

    # Leyenda de categorias
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
               markersize=10, label='PBH'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Estelar'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='IMBH'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=10, label='SMBH'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='UMBH'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Parametros MCV por Categoria de Agujero Negro', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {save_path}")

    return fig


def figura2_perfiles_radiales(save_path: str = None):
    """
    Figura 2: Perfiles radiales de MCV para BH canonicos.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # BH canonicos a analizar
    bh_nombres = ['Cygnus_X1', 'SgrA', 'M87', 'TON618']
    colores = ['blue', 'green', 'orange', 'red']

    # Rango radial
    r_rs = np.logspace(0, 3, 200)

    for idx, (nombre, color) in enumerate(zip(bh_nombres, colores)):
        bh = crear_BH_canonico(nombre)
        r = r_rs * bh.r_s

        # Calcular perfiles
        rho = bh.rho_MCV(r)
        Xi = bh.Xi(r)
        S = bh.S_local(r)
        dt = bh.factor_tiempo_relativo(r)

        # Panel 1: rho_MCV
        axes[0, 0].loglog(r_rs, rho, color=color, label=bh.nombre, linewidth=2)

        # Panel 2: Xi
        axes[0, 1].loglog(r_rs, Xi, color=color, label=bh.nombre, linewidth=2)

        # Panel 3: S_local
        axes[1, 0].semilogx(r_rs, S, color=color, label=bh.nombre, linewidth=2)

        # Panel 4: Dilatacion temporal
        axes[1, 1].loglog(r_rs, dt, color=color, label=bh.nombre, linewidth=2)

    # Configurar paneles
    axes[0, 0].set_xlabel('r/r$_s$')
    axes[0, 0].set_ylabel('$\\rho_{MCV}$ [kg/m$^3$]')
    axes[0, 0].set_title('Densidad de MCV')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('r/r$_s$')
    axes[0, 1].set_ylabel('$\\Xi$')
    axes[0, 1].set_title('Potencial Cronologico')
    axes[0, 1].axhline(y=XI_FREEZE, color='gray', linestyle='--', alpha=0.7,
                       label=f'$\\Xi_{{freeze}}$ = {XI_FREEZE}')
    axes[0, 1].axhline(y=XI_BUBBLE, color='gray', linestyle=':', alpha=0.7,
                       label=f'$\\Xi_{{bubble}}$ = {XI_BUBBLE}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('r/r$_s$')
    axes[1, 0].set_ylabel('S$_{local}$')
    axes[1, 0].set_title('Entropia Local')
    axes[1, 0].axhline(y=S_EXT, color='gray', linestyle='--', alpha=0.7,
                       label=f'S$_{{ext}}$ = {S_EXT}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].set_xlabel('r/r$_s$')
    axes[1, 1].set_ylabel('$\Delta$t/$\Delta$t$_0$')
    axes[1, 1].set_title('Dilatacion Temporal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Perfiles Radiales MCV para BH Canonicos', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {save_path}")

    return fig


def figura3_condiciones_Slocal(save_path: str = None):
    """
    Figura 3: Condiciones para diferentes valores de S_local.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Usar Sgr A* como ejemplo
    bh = crear_BH_canonico('SgrA')

    # Valores de S_local a marcar
    S_targets = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85]
    colors_S = plt.cm.viridis(np.linspace(0, 0.9, len(S_targets)))

    # Rango radial
    r_rs = np.logspace(0, 2.5, 200)
    r = r_rs * bh.r_s

    # Calcular perfiles
    S_array = bh.S_local(r)
    Xi_array = bh.Xi(r)
    rho_array = bh.rho_MCV(r)
    dt_array = bh.factor_tiempo_relativo(r)

    # Panel 1: S_local vs r/r_s con puntos marcados
    ax1 = axes[0, 0]
    ax1.semilogx(r_rs, S_array, 'b-', linewidth=2)
    ax1.axhline(y=S_EXT, color='gray', linestyle='--', label=f'S$_{{ext}}$ = {S_EXT}')

    for S_target, color in zip(S_targets, colors_S):
        idx = np.argmin(np.abs(S_array - S_target))
        ax1.scatter(r_rs[idx], S_array[idx], color=color, s=100, zorder=5,
                   edgecolors='black', linewidth=1)
        ax1.annotate(f'S={S_target:.2f}', (r_rs[idx], S_array[idx]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax1.set_xlabel('r/r$_s$')
    ax1.set_ylabel('S$_{local}$')
    ax1.set_title('Entropia Local vs Radio (Sgr A*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 1)

    # Panel 2: Xi vs S_local
    ax2 = axes[0, 1]
    ax2.semilogy(S_array, Xi_array, 'r-', linewidth=2)
    for S_target, color in zip(S_targets, colors_S):
        idx = np.argmin(np.abs(S_array - S_target))
        ax2.scatter(S_array[idx], Xi_array[idx], color=color, s=100, zorder=5,
                   edgecolors='black', linewidth=1)

    ax2.axhline(y=XI_FREEZE, color='blue', linestyle='--', alpha=0.7,
               label=f'$\\Xi_{{freeze}}$ = {XI_FREEZE}')
    ax2.axhline(y=XI_BUBBLE, color='green', linestyle=':', alpha=0.7,
               label=f'$\\Xi_{{bubble}}$ = {XI_BUBBLE}')
    ax2.set_xlabel('S$_{local}$')
    ax2.set_ylabel('$\\Xi$')
    ax2.set_title('Potencial Cronologico vs Entropia Local')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Dilatacion vs S_local
    ax3 = axes[1, 0]
    ax3.semilogy(S_array, dt_array, 'g-', linewidth=2)
    for S_target, color in zip(S_targets, colors_S):
        idx = np.argmin(np.abs(S_array - S_target))
        ax3.scatter(S_array[idx], dt_array[idx], color=color, s=100, zorder=5,
                   edgecolors='black', linewidth=1)

    ax3.set_xlabel('S$_{local}$')
    ax3.set_ylabel('$\Delta$t/$\Delta$t$_0$')
    ax3.set_title('Dilatacion Temporal vs Entropia Local')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Tabla de condiciones
    ax4 = axes[1, 1]
    ax4.axis('off')

    tabla = bh.tabla_S_local(S_targets)
    cell_text = []
    for fila in tabla:
        cell_text.append([
            f"{fila['S_local']:.2f}",
            f"{fila['Delta_S']:.2f}",
            f"{fila['r_rs']:.2f}",
            f"{fila['r_m']:.2e}",
            f"{fila['Xi']:.3f}",
            f"{fila['dilatacion']:.2f}",
        ])

    col_labels = ['S$_{local}$', '$\Delta$S', 'r/r$_s$', 'r [m]', '$\\Xi$', '$\Delta$t/$\Delta$t$_0$']

    table = ax4.table(cellText=cell_text, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Tabla de Condiciones (Sgr A*)', pad=20)

    plt.suptitle('Condiciones para Diferentes Valores de S$_{local}$', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {save_path}")

    return fig


def figura4_burbuja_entropica(save_path: str = None):
    """
    Figura 4: Definicion y estructura de la burbuja entropica.
    """
    fig = plt.figure(figsize=(16, 10))

    # Crear grid para subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Diagrama esquematico de la burbuja (grande)
    ax1 = fig.add_subplot(gs[0, :2])

    # Dibujar zonas concentricas
    radii = [1, 1.5, 3, 20, 100]
    labels = ['Horizonte\n(BH)', 'Colapso\nDimensional', 'Congelacion',
              'Burbuja', 'Exterior']
    colors_zones = ['black', 'darkred', 'red', 'orange', 'lightblue']
    alphas = [1.0, 0.8, 0.6, 0.4, 0.2]

    # Escala logaritmica para visualizacion
    log_radii = np.log10(np.array(radii) + 0.1)
    max_r = log_radii[-1] * 1.1

    for i in range(len(radii) - 1, -1, -1):
        circle = plt.Circle((0, 0), log_radii[i], color=colors_zones[i],
                            alpha=alphas[i], zorder=i)
        ax1.add_patch(circle)

    # Etiquetas
    ax1.annotate('Horizonte\nr = r$_s$', (0, 0), fontsize=10, ha='center',
                color='white', fontweight='bold')
    ax1.annotate('$\\Xi$ > 10\nColapso D', (0, log_radii[1] * 0.7), fontsize=9,
                ha='center', color='white')
    ax1.annotate('1 < $\\Xi$ < 10\nCongelacion', (0, log_radii[2] * 0.8), fontsize=9,
                ha='center')
    ax1.annotate('0.1 < $\\Xi$ < 1\nBurbuja', (0, log_radii[3] * 0.8), fontsize=9,
                ha='center')
    ax1.annotate('$\\Xi$ < 0.1\nExterior', (log_radii[4] * 0.7, 0), fontsize=9,
                ha='center')

    ax1.set_xlim(-max_r, max_r)
    ax1.set_ylim(-max_r, max_r)
    ax1.set_aspect('equal')
    ax1.set_title('Estructura de la Burbuja Entropica', fontsize=14)
    ax1.axis('off')

    # Panel 2: Leyenda de umbrales
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    leyenda_text = """
    UMBRALES ONTOLOGICOS

    r_s: Radio de Schwarzschild
         r_s = 2GM/c^2

    r_collapse: Degradacion dimensional
         Xi = 10

    r_freeze: Congelacion temporal
         Xi = 1.0

    r_bubble: Limite de burbuja
         Xi = 0.1

    CONDICIONES:
    - S_local < S_ext (friccion)
    - rho_MCV > rho_crit (activacion)
    - Xi > 0.1 (aislamiento)
    - dtau/dt < 1 (tiempo lento)
    """
    ax2.text(0.1, 0.9, leyenda_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Perfil de Xi con zonas marcadas
    ax3 = fig.add_subplot(gs[1, 0])

    bh = crear_BH_canonico('SgrA')
    r_rs = np.logspace(0, 2.5, 200)
    r = r_rs * bh.r_s
    Xi = bh.Xi(r)

    ax3.loglog(r_rs, Xi, 'b-', linewidth=2)

    # Marcar zonas
    ax3.axhline(y=XI_COLLAPSE, color='darkred', linestyle='--',
               label=f'$\\Xi_{{collapse}}$ = {XI_COLLAPSE}')
    ax3.axhline(y=XI_FREEZE, color='red', linestyle='--',
               label=f'$\\Xi_{{freeze}}$ = {XI_FREEZE}')
    ax3.axhline(y=XI_BUBBLE, color='orange', linestyle='--',
               label=f'$\\Xi_{{bubble}}$ = {XI_BUBBLE}')

    # Sombrear zonas
    ax3.axhspan(XI_COLLAPSE, 1000, alpha=0.2, color='darkred')
    ax3.axhspan(XI_FREEZE, XI_COLLAPSE, alpha=0.2, color='red')
    ax3.axhspan(XI_BUBBLE, XI_FREEZE, alpha=0.2, color='orange')

    ax3.set_xlabel('r/r$_s$')
    ax3.set_ylabel('$\\Xi$')
    ax3.set_title('Potencial Cronologico (Sgr A*)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.01, 100)

    # Panel 4: Dilatacion temporal
    ax4 = fig.add_subplot(gs[1, 1])

    dt = bh.factor_tiempo_relativo(r)
    ax4.loglog(r_rs, dt, 'g-', linewidth=2)

    ax4.set_xlabel('r/r$_s$')
    ax4.set_ylabel('$\Delta$t/$\Delta$t$_0$')
    ax4.set_title('Dilatacion Temporal')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Comparacion de radios por categoria
    ax5 = fig.add_subplot(gs[1, 2])

    categorias = ['Stellar', 'IMBH', 'SMBH', 'UMBH']
    r_freeze_vals = []
    r_bubble_vals = []

    for cat in categorias:
        if cat == 'Stellar':
            bh_cat = crear_BH_canonico('Cygnus_X1')
        elif cat == 'IMBH':
            bh_cat = MCV_AgujerosNegros(1e4, 'IMBH')
        elif cat == 'SMBH':
            bh_cat = crear_BH_canonico('SgrA')
        else:
            bh_cat = crear_BH_canonico('TON618')

        radios = bh_cat.calcular_radios_caracteristicos()
        r_freeze_vals.append(radios['r_freeze_rs'])
        r_bubble_vals.append(radios['r_bubble_rs'])

    x = np.arange(len(categorias))
    width = 0.35

    bars1 = ax5.bar(x - width/2, r_freeze_vals, width, label='r$_{freeze}$/r$_s$',
                   color='red', alpha=0.7)
    bars2 = ax5.bar(x + width/2, r_bubble_vals, width, label='r$_{bubble}$/r$_s$',
                   color='orange', alpha=0.7)

    ax5.set_xlabel('Categoria de BH')
    ax5.set_ylabel('Radio / r$_s$')
    ax5.set_title('Radios Caracteristicos')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categorias, rotation=45)
    ax5.legend()
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Definicion de la Burbuja Entropica', fontsize=16, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {save_path}")

    return fig


def guardar_resultados_json(output_path: str = None):
    """
    Guarda los resultados numericos en formato JSON.
    """
    if output_path is None:
        output_path = 'mcv_bh_analysis_results.json'

    resultados = {
        'categorias': analizar_por_categorias(),
        'ejemplos_canonicos': {},
        'sellos_ontologicos': [
            {'nombre': s.nombre, 'S_n': s.S_n, 'energia_GeV': s.energia_GeV,
             'proceso': s.proceso} for s in SELLOS
        ],
        'umbrales': {
            'XI_BUBBLE': XI_BUBBLE,
            'XI_FREEZE': XI_FREEZE,
            'XI_COLLAPSE': XI_COLLAPSE,
            'S_EXT': S_EXT,
        }
    }

    # Ejemplos canonicos
    for nombre in EJEMPLOS_CANONICOS.keys():
        bh = crear_BH_canonico(nombre)
        resultados['ejemplos_canonicos'][nombre] = bh.analisis_completo()

    # Convertir numpy a float para JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    resultados = convert_numpy(resultados)

    with open(output_path, 'w') as f:
        json.dump(resultados, f, indent=2)

    print(f"  Resultados guardados: {output_path}")
    return resultados


def generar_todas_figuras(output_dir: str = '.'):
    """
    Genera todas las figuras del analisis.

    Args:
        output_dir: Directorio de salida
    """
    print("\n" + "="*70)
    print("  GENERANDO VISUALIZACIONES MCV-BH")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Figura 1: Parametros por masa...")
    figura1_parametros_masa(os.path.join(output_dir, 'figura1_parametros_masa.png'))

    print("\n[2] Figura 2: Perfiles radiales...")
    figura2_perfiles_radiales(os.path.join(output_dir, 'figura2_perfiles_radiales.png'))

    print("\n[3] Figura 3: Condiciones S_local...")
    figura3_condiciones_Slocal(os.path.join(output_dir, 'figura3_condiciones_Slocal.png'))

    print("\n[4] Figura 4: Burbuja entropica...")
    figura4_burbuja_entropica(os.path.join(output_dir, 'figura4_burbuja_entropica.png'))

    print("\n[5] Guardando resultados JSON...")
    guardar_resultados_json(os.path.join(output_dir, 'mcv_bh_analysis_results.json'))

    print("\n" + "="*70)
    print("  VISUALIZACIONES COMPLETADAS")
    print("="*70)


if __name__ == "__main__":
    generar_todas_figuras()
    plt.show()
