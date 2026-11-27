import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
import fabio
import re
import io
import os
import json
import tempfile
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="结构生物学数据分析中心", layout="wide", page_icon="🧬")


# =====================================================================
#                     模块 1: Table 1 与残基/3D 口袋分析引擎
# =====================================================================
KD_HYDROPHOBICITY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}
AA_VARIANT_MAP = {
    'MSE': 'MET', 'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'CYX': 'CYS', 'ASH': 'ASP', 'GLH': 'GLU'
}
STANDARD_AA = set(KD_HYDROPHOBICITY.keys()) | set(AA_VARIANT_MAP.keys())
MAIN_CHAIN_ATOMS = {'N', 'CA', 'C', 'O'}
WATER_RESNAMES = {'HOH', 'WAT', 'DOD', 'TIP', 'TIP3', 'SOL'}


def get_kd_hydro(resname):
    canonical = AA_VARIANT_MAP.get(resname, resname)
    return KD_HYDROPHOBICITY.get(canonical, 0.0)


def parse_aimless_log(content):
    data = {}
    summary_match = re.search(r'<!--SUMMARY_BEGIN-->(.*?)<!--SUMMARY_END-->', content, re.DOTALL)
    summary = summary_match.group(1) if summary_match else content

    def get_vals(pattern, source=summary):
        match = re.search(pattern, source)
        return match.groups() if match else (None, None, None)

    m_cell = re.search(
        r'Average unit cell:\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)',
        content
    )
    if m_cell:
        data['a'], data['b'], data['c'] = m_cell.group(1), m_cell.group(2), m_cell.group(3)
        data['alpha'], data['beta'], data['gamma'] = m_cell.group(4), m_cell.group(5), m_cell.group(6)

    m_sg = re.search(r'Space group[^\n]*:\s*(.*)', content)
    if m_sg:
        data['space_group'] = m_sg.group(1).strip()

    m_wave = re.search(r'Wavelength:\s+([\d\.]+)', content)
    if m_wave:
        data['wavelength'] = m_wave.group(1)

    m_low = get_vals(r'Low resolution limit\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_low[0]:
        data['low_res'] = m_low[0]

    m_high = get_vals(r'High resolution limit\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_high[0]:
        data['high_res'], data['high_res_outer'] = m_high[0], m_high[2]

    m_tot = get_vals(r'Total number of observations\s+(\d+)\s+(\d+)\s+(\d+)')
    if m_tot[0]:
        data['total_refl_overall'], data['total_refl_outer'] = m_tot[0], m_tot[2]

    m_uniq = get_vals(r'Total number unique\s+(\d+)\s+(\d+)\s+(\d+)')
    if m_uniq[0]:
        data['unique_refl_overall'], data['unique_refl_outer'] = m_uniq[0], m_uniq[2]

    m_rmerge = get_vals(r'Rmerge\s*\(all I\+\s*(?:and|&)\s*I-\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_rmerge[0]:
        data['rmerge_overall'], data['rmerge_outer'] = m_rmerge[0], m_rmerge[2]

    m_rpim = get_vals(r'Rpim\s*\(all I\+\s*(?:and|&)\s*I-\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if not m_rpim[0]:
        m_rpim = get_vals(r'Rpim\s*\([^\)]*\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if not m_rpim[0]:
        m_rpim = get_vals(r'Rpim\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_rpim[0]:
        data['rpim_overall'], data['rpim_outer'] = m_rpim[0], m_rpim[2]

    m_cc12 = get_vals(r'Mn\(I\) half-set correlation CC\(1/2\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if not m_cc12[0]:
        m_cc12 = get_vals(r'CC\(1/2\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_cc12[0]:
        data['cc12_overall'], data['cc12_outer'] = m_cc12[0], m_cc12[2]

    m_isigi = get_vals(r'Mean\(\(I\)/sd\(I\)\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if not m_isigi[0]:
        m_isigi = get_vals(r'Mean I/sigI\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_isigi[0]:
        data['isigi_overall'], data['isigi_outer'] = m_isigi[0], m_isigi[2]

    m_comp = get_vals(r'Completeness\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_comp[0]:
        data['comp_overall'], data['comp_outer'] = m_comp[0], m_comp[2]

    m_mult = get_vals(r'Multiplicity\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_mult[0]:
        data['mult_overall'], data['mult_outer'] = m_mult[0], m_mult[2]

    m_wilson = re.search(
        r'(?:Wilson\s+B[- ]?factor|B[- ]?factor\s+from\s+Wilson\s+plot|Estimate\s+of\s+Wilson\s+B)[^:=\n]*[:=]\s*([\d\.]+)',
        content, re.IGNORECASE
    )
    if m_wilson:
        data['wilson_b'] = f"{float(m_wilson.group(1)):.2f}"

    return data


def parse_scale_log(content):
    data = {}
    m_cell = re.findall(
        r'Cell,\s*a\s+([\d\.]+)\s+b\s+([\d\.]+)\s+c\s+([\d\.]+)\s+alpha\s+([\d\.]+)\s+beta\s+([\d\.]+)\s+gamma\s+([\d\.]+)',
        content
    )
    if m_cell:
        data['a'], data['b'], data['c'] = m_cell[-1][0], m_cell[-1][1], m_cell[-1][2]
        data['alpha'], data['beta'], data['gamma'] = m_cell[-1][3], m_cell[-1][4], m_cell[-1][5]

    m_sg = re.search(r'Space group\s+(.*?)\s+number', content, re.IGNORECASE)
    if m_sg:
        data['space_group'] = m_sg.group(1).strip()

    lines = content.split('\n')
    in_r_table, r_table = False, []
    for line in lines:
        if "Summary of reflections intensities and R-factors by shells" in line:
            in_r_table, r_table = True, []
        elif in_r_table:
            if not line.strip() and len(r_table) > 5:
                in_r_table = False
            else:
                r_table.append(line.strip())

    if r_table:
        data_lines = [l.split() for l in r_table if l and l[0].replace('.', '', 1).isdigit()]
        if data_lines:
            outer = data_lines[-1]
            data['low_res'], data['high_res'], data['high_res_outer'] = data_lines[0][0], outer[1], outer[1]
            data['rmerge_outer'] = outer[6]
            try:
                data['isigi_outer'] = f"{float(outer[2]) / float(outer[3]):.1f}"
            except Exception:
                pass
            if len(outer) > 9:
                data['rpim_outer'] = outer[9]
            if len(outer) > 10:
                data['cc12_outer'] = outer[10]

        for l in r_table:
            if l.startswith("All reflections") or l.startswith("All hkl"):
                parts = l.split()
                data['rmerge_overall'] = parts[6]
                try:
                    data['isigi_overall'] = f"{float(parts[2]) / float(parts[3]):.1f}"
                except Exception:
                    pass
                if len(parts) > 9:
                    data['rpim_overall'] = parts[9]
                if len(parts) > 10:
                    data['cc12_overall'] = parts[10]

    in_comp, comp_table = False, []
    for line in lines:
        if "Summary of observation redundancies:" in line:
            in_comp, comp_table = True, []
        elif in_comp:
            if not line.strip() and len(comp_table) > 5:
                in_comp = False
            else:
                comp_table.append(line.strip())

    if comp_table:
        data_lines = [l.split() for l in comp_table if l and l[0].replace('.', '', 1).isdigit()]
        if data_lines:
            data['comp_outer'] = data_lines[-1][-1]
            if len(data_lines[-1]) >= 3 and data_lines[-1][2].isdigit():
                data['unique_refl_outer'] = data_lines[-1][2]
        for l in comp_table:
            if l.startswith("All hkl"):
                parts = l.split()
                data['comp_overall'] = parts[-1]
                if len(parts) >= 3 and parts[2].isdigit():
                    data['unique_refl_overall'] = parts[2]

    in_red, red_table = False, []
    for line in lines:
        if "Average Redundancy Per Shell" in line:
            in_red, red_table = True, []
        elif in_red:
            if not line.strip() and len(red_table) > 3:
                in_red = False
            else:
                red_table.append(line.strip())

    if red_table:
        data_lines = [l.split() for l in red_table if l and l[0].replace('.', '', 1).isdigit()]
        if data_lines:
            data['mult_outer'] = data_lines[-1][-1]
        for l in red_table:
            if l.startswith("All hkl"):
                data['mult_overall'] = l.split()[-1]

    return data


def parse_pdb(content):
    data = {}
    if not content:
        return data

    prot_b, lig_b, wat_b = [], [], []
    lig_resnames = set()

    for line in content.splitlines():
        if line.startswith("CRYST1"):
            data['space_group'] = line[55:66].strip()
            data['a'], data['b'], data['c'] = line[6:15].strip(), line[15:24].strip(), line[24:33].strip()
            data['alpha'], data['beta'], data['gamma'] = line[33:40].strip(), line[40:47].strip(), line[47:54].strip()
        elif line.startswith("REMARK   3"):
            m_bond = re.search(r'BOND\s+:\s+([\d\.]+)', line)
            if m_bond:
                data['rmsd_bonds'] = m_bond.group(1)
            m_angle = re.search(r'ANGLE\s+:\s+([\d\.]+)', line)
            if m_angle:
                data['rmsd_angles'] = m_angle.group(1)
            if 'NUMBER OF REFLECTIONS' in line and 'NON-ANOMALOUS' not in line:
                m_tot = re.search(r':\s*(\d+)', line)
                if m_tot:
                    data['refl_total'] = m_tot.group(1)
            if 'FREE R VALUE TEST SET COUNT' in line or 'NUMBER OF REFLECTIONS IN TEST SET' in line:
                m_free = re.search(r':\s*(\d+)', line)
                if m_free:
                    data['refl_free'] = m_free.group(1)
            if 'FAVORED' in line and '%' in line:
                m_fav = re.search(r':\s*([\d\.]+)', line)
                if m_fav:
                    data['rama_favored'] = m_fav.group(1)
            if 'ALLOWED' in line and '%' in line:
                m_all = re.search(r':\s*([\d\.]+)', line)
                if m_all:
                    data['rama_allowed'] = m_all.group(1)
            if 'OUTLIERS' in line and '%' in line and 'ROTAMER' not in line:
                m_out = re.search(r':\s*([\d\.]+)', line)
                if m_out:
                    data['rama_outlier'] = m_out.group(1)
            if 'WILSON' in line.upper():
                m_wb = re.search(r':\s*([\d\.]+)', line)
                if m_wb:
                    data['wilson_b'] = f"{float(m_wb.group(1)):.2f}"
            if 'CLASHSCORE' in line.upper():
                m_clash = re.search(r':\s*([\d\.]+)', line)
                if m_clash:
                    data['clashscore'] = m_clash.group(1)
            if 'ROTAMER OUTLIERS' in line.upper():
                m_rot = re.search(r':\s*([\d\.]+)', line)
                if m_rot:
                    data['rotamer_outliers'] = m_rot.group(1)
            if 'MEAN B VALUE' in line.upper():
                m_mb = re.search(r':\s*([\d\.]+)', line)
                if m_mb:
                    data['b_all'] = f"{float(m_mb.group(1)):.2f}"
        elif line.startswith(("ATOM", "HETATM")):
            try:
                resname = line[17:20].strip()
                bf = float(line[60:66])
                if resname in WATER_RESNAMES:
                    wat_b.append(bf)
                elif line.startswith("ATOM") and resname in STANDARD_AA:
                    prot_b.append(bf)
                else:
                    lig_b.append(bf)
                    lig_resnames.add(resname)
            except ValueError:
                pass

    if 'refl_total' in data and 'refl_free' in data:
        data['refl_work'] = str(int(data['refl_total']) - int(data['refl_free']))

    all_b = prot_b + lig_b + wat_b
    if all_b and 'b_all' not in data:
        data['b_all'] = f"{np.mean(all_b):.2f}"
    if prot_b:
        data['b_protein'] = f"{np.mean(prot_b):.2f}"
        data['protein_atoms'] = str(len(prot_b))
    if lig_b:
        data['b_ligand'] = f"{np.mean(lig_b):.2f}"
        data['ligand_atoms'] = str(len(lig_b))
        data['ligand_names'] = ", ".join(sorted(lig_resnames))
    if wat_b:
        data['b_water'] = f"{np.mean(wat_b):.2f}"
        data['water_atoms'] = str(len(wat_b))

    return data


def parse_phenix_log(content):
    data = {}
    m_r = re.search(r'Final R-work\s*=\s*([\d\.]+),\s*R-free\s*=\s*([\d\.]+)', content)
    if m_r:
        data['r_work'], data['r_free'] = m_r.group(1), m_r.group(2)

    m_res = re.findall(r'resolution:\s*([\d\.]+)\s*-\s*([\d\.]+)\s*[AÅ],\s*n_refl\.=(\d+)', content)
    if m_res:
        r1, r2 = float(m_res[-1][0]), float(m_res[-1][1])
        data['refine_res_high'], data['refine_res_low'] = f"{min(r1, r2):.2f}", f"{max(r1, r2):.2f}"

    m_prot = re.findall(r'Protein:\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)\s+[^\s]+\s+(\d+)', content)
    if m_prot:
        data['b_protein'], data['protein_atoms'] = m_prot[-1][0], m_prot[-1][1]

    m_lig = re.findall(
        r'(?:Ligands?(?:/ions?)?|Non-polymer|Non-water|Other|Heterogen|Small\s+molecules?|Ions?)(?:\s*\([^\)]*\))?\s*:\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)\s+[^\s]+\s+(\d+)',
        content, re.IGNORECASE
    )
    if m_lig:
        data['b_ligand'], data['ligand_atoms'] = m_lig[-1][0], m_lig[-1][1]

    m_wat = re.findall(r'(?:Water|Solvent):\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)\s+[^\s]+\s+(\d+)', content, re.IGNORECASE)
    if m_wat:
        data['b_water'], data['water_atoms'] = m_wat[-1][0], m_wat[-1][1]

    m_ball = re.findall(r'All:\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)', content)
    if m_ball:
        data['b_all'] = m_ball[-1]

    m_wilson = re.findall(
        r'(?:Wilson\s+B(?:-?\s*factor)?|FROM\s+WILSON\s+PLOT|B_wilson)[^:=\n]*[:=]\s*([\d\.]+)',
        content, re.IGNORECASE
    )
    if m_wilson:
        data['wilson_b'] = f"{float(m_wilson[-1]):.2f}"

    m_clash = re.findall(r'(?:All-atom\s+)?clashscore\s*[:=]\s*([\d\.]+)', content, re.IGNORECASE)
    if m_clash:
        data['clashscore'] = m_clash[-1]

    m_rot = re.findall(r'Rotamer\s+outliers\s*[:=]\s*([\d\.]+)', content, re.IGNORECASE)
    if m_rot:
        data['rotamer_outliers'] = m_rot[-1]

    m_rama = re.findall(
        r'Ramachandran\s+plot:.*?outliers\s*[:=]\s*([\d\.]+)%.*?allowed\s*[:=]\s*([\d\.]+)%.*?favored\s*[:=]\s*([\d\.]+)%',
        content, re.DOTALL | re.IGNORECASE
    )
    if m_rama:
        data['rama_outlier'], data['rama_allowed'], data['rama_favored'] = m_rama[-1]

    return data


def build_table_data(dcoll, pdb, phenix, custom_wavelength="0.9792"):
    def br(v1, v2):
        return f"{v1} ({v2})" if v1 and v2 else v1 or "N/A"

    r_work_val = float(phenix['r_work']) if 'r_work' in phenix else None
    r_free_val = float(phenix['r_free']) if 'r_free' in phenix else None
    r_work = f"{(r_work_val * 100 if r_work_val <= 1.0 else r_work_val):.2f}" if r_work_val is not None else 'N/A'
    r_free = f"{(r_free_val * 100 if r_free_val <= 1.0 else r_free_val):.2f}" if r_free_val is not None else 'N/A'

    rama_fav = pdb.get('rama_favored') or phenix.get('rama_favored')
    rama_all = pdb.get('rama_allowed') or phenix.get('rama_allowed')
    rama_out = pdb.get('rama_outlier') or phenix.get('rama_outlier')
    rama = f"{rama_fav}/{rama_all or '0'}/{rama_out or '0'}" if rama_fav else "N/A"

    refl = f"{pdb['refl_work']}/{pdb['refl_free']}" if 'refl_work' in pdb and 'refl_free' in pdb else "N/A"

    a = pdb.get('a') or dcoll.get('a') or 'N/A'
    b = pdb.get('b') or dcoll.get('b') or 'N/A'
    c = pdb.get('c') or dcoll.get('c') or 'N/A'
    alpha = pdb.get('alpha') or dcoll.get('alpha') or 'N/A'
    beta = pdb.get('beta') or dcoll.get('beta') or 'N/A'
    gamma = pdb.get('gamma') or dcoll.get('gamma') or 'N/A'
    sg = pdb.get('space_group') or dcoll.get('space_group') or 'N/A'
    final_wavelength = dcoll.get('wavelength') or custom_wavelength

    wilson_b = dcoll.get('wilson_b') or phenix.get('wilson_b') or pdb.get('wilson_b') or 'N/A'
    b_all = phenix.get('b_all') or pdb.get('b_all') or 'N/A'
    b_protein = phenix.get('b_protein') or pdb.get('b_protein') or 'N/A'
    b_ligand = phenix.get('b_ligand') or pdb.get('b_ligand')
    b_water = phenix.get('b_water') or pdb.get('b_water') or 'N/A'

    protein_atoms = phenix.get('protein_atoms') or pdb.get('protein_atoms') or 'N/A'
    ligand_atoms = phenix.get('ligand_atoms') or pdb.get('ligand_atoms')
    water_atoms = phenix.get('water_atoms') or pdb.get('water_atoms') or 'N/A'

    clashscore = phenix.get('clashscore') or pdb.get('clashscore') or 'N/A'
    rotamer_out = phenix.get('rotamer_outliers') or pdb.get('rotamer_outliers') or 'N/A'

    rows = [
        ["Data Collection", ""],
        ["Wavelength (Å)", final_wavelength],
        ["Space group", sg],
        ["Cell parameters", ""],
        ["a, b, c (Å)", f"{a}, {b}, {c}"],
        ["α, β, γ (°)", f"{alpha}, {beta}, {gamma}"],
        ["Resolution (Å)", f"{dcoll.get('low_res', 'N/A')}–{dcoll.get('high_res', 'N/A')}\n({dcoll.get('high_res', 'N/A')}–{dcoll.get('high_res_outer', 'N/A')})"],
        ["Total reflections", br(dcoll.get('total_refl_overall'), dcoll.get('total_refl_outer'))],
        ["Unique reflections", br(dcoll.get('unique_refl_overall'), dcoll.get('unique_refl_outer'))],
        ["Rmerge (%)", br(dcoll.get('rmerge_overall'), dcoll.get('rmerge_outer'))],
        ["Rpim (%)", br(dcoll.get('rpim_overall'), dcoll.get('rpim_outer'))],
        ["CC1/2", br(dcoll.get('cc12_overall'), dcoll.get('cc12_outer'))],
        ["I/σI", br(dcoll.get('isigi_overall'), dcoll.get('isigi_outer'))],
        ["Completeness (%)", br(dcoll.get('comp_overall'), dcoll.get('comp_outer'))],
        ["Redundancy", br(dcoll.get('mult_overall'), dcoll.get('mult_outer'))],
        ["Wilson B-factor (Å²)", wilson_b],
        ["Refinement", ""],
        ["Resolution (Å)", f"{phenix.get('refine_res_low', 'N/A')}–{phenix.get('refine_res_high', 'N/A')}"],
        ["No. reflections used/free", refl],
        ["Rwork / Rfree (%)", f"{r_work}/{r_free}"],
        ["No. atoms", ""],
        ["Protein", protein_atoms],
    ]

    if ligand_atoms and str(ligand_atoms) != '0':
        rows.append(["Ligand / Ion", ligand_atoms])

    rows.extend([
        ["Water", water_atoms],
        ["B-factors (Å²)", ""],
        ["Average", b_all],
        ["Protein", b_protein],
    ])

    if b_ligand:
        rows.append(["Ligand / Ion", b_ligand])

    rows.extend([
        ["Water", b_water],
        ["R.m.s. deviations", ""],
        ["Bond lengths (Å)", pdb.get('rmsd_bonds', 'N/A')],
        ["Bond angles (°)", pdb.get('rmsd_angles', 'N/A')],
        ["Validation", ""],
        ["Ramachandran favored/allowed/outlier (%)", rama],
        ["Rotamer outliers (%)", rotamer_out],
        ["Clashscore", clashscore]
    ])

    return rows


@st.cache_data
def draw_table_to_buffer(table_content):
    df = pd.DataFrame(table_content, columns=["Parameter", "Value"])
    fig, ax = plt.subplots(figsize=(8.5, len(df) * 0.34 + 1.0))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.75)

    top_level_labels = {
        "Data Collection", "Wavelength (Å)", "Space group", "Cell parameters",
        "Resolution (Å)", "Total reflections", "Unique reflections", "Rmerge (%)",
        "Rpim (%)", "CC1/2", "I/σI", "Completeness (%)", "Redundancy",
        "Wilson B-factor (Å²)", "Refinement", "No. reflections used/free",
        "Rwork / Rfree (%)", "No. atoms", "B-factors (Å²)", "R.m.s. deviations",
        "Validation"
    }

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('white')
        text = cell.get_text().get_text()
        if text in ["Data Collection", "Refinement"]:
            cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.visible_edges = 'TB'
            cell.set_linewidth(1.5)
        if col == 0 and row > 0 and text not in top_level_labels:
            cell.get_text().set_text("    " + text)
        if col == 0:
            cell.PAD = 0.05

    for col in range(2):
        table.get_celld()[(0, col)].set_edgecolor('black')
        table.get_celld()[(0, col)].visible_edges = 'T'
        table.get_celld()[(0, col)].set_linewidth(1.5)
        table.get_celld()[(len(df) - 1, col)].set_edgecolor('black')
        table.get_celld()[(len(df) - 1, col)].visible_edges = 'B'
        table.get_celld()[(len(df) - 1, col)].set_linewidth(1.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# --- 蛋白残基 2D 与 3D 分析 ---
def parse_pdb_for_profile(content):
    atom_records = []
    ligands = {}
    for line in content.splitlines():
        record_type = line[:6].strip()
        if record_type not in ("ATOM", "HETATM"):
            continue
        try:
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            if resname in WATER_RESNAMES:
                continue
            chain = line[21].strip() or "A"
            resseq = int(line[22:26].strip())
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            bfactor = float(line[60:66])

            if record_type == "ATOM" and resname in STANDARD_AA:
                atom_records.append({
                    'chain': chain, 'resseq': resseq, 'resname': resname,
                    'is_mc': atom_name in MAIN_CHAIN_ATOMS,
                    'x': x, 'y': y, 'z': z, 'bfactor': bfactor
                })
            else:
                lig_key = f"{resname} (Chain {chain}, {resseq})"
                ligands.setdefault(lig_key, []).append([x, y, z])
        except ValueError:
            continue

    df_atoms = pd.DataFrame(atom_records)
    return df_atoms, {k: np.array(v) for k, v in ligands.items()}


def build_residue_profile(df_atoms, chain_id, lig_coords=None, dist_cutoff=4.5, window_size=5):
    df_c = df_atoms[df_atoms['chain'] == chain_id].copy()
    if df_c.empty:
        return pd.DataFrame()

    if lig_coords is not None and len(lig_coords) > 0:
        coords = df_c[['x', 'y', 'z']].values
        df_c['min_dist'] = cdist(coords, lig_coords).min(axis=1)
    else:
        df_c['min_dist'] = np.nan

    grouped = []
    for (resseq, resname), grp in df_c.groupby(['resseq', 'resname'], sort=True):
        mc_b = grp[grp['is_mc']]['bfactor'].mean()
        sc_b = grp[~grp['is_mc']]['bfactor'].mean()
        all_b = grp['bfactor'].mean()
        min_d = grp['min_dist'].min() if not grp['min_dist'].isna().all() else np.nan
        grouped.append({
            'resseq': resseq,
            'resname': resname,
            'b_all': round(all_b, 2),
            'b_mc': round(mc_b, 2),
            'b_sc': round(sc_b if not np.isnan(sc_b) else mc_b, 2),
            'hydro_raw': get_kd_hydro(resname),
            'min_dist': round(min_d, 2) if not np.isnan(min_d) else np.nan,
            'in_pocket': bool(min_d <= dist_cutoff) if not np.isnan(min_d) else False
        })

    df_res = pd.DataFrame(grouped)
    df_res['hydro_smooth'] = df_res['hydro_raw'].rolling(window=window_size, center=True, min_periods=1).mean()
    return df_res


def draw_residue_profile(df_res, chain_id, selected_lig=None, dist_cutoff=4.5):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    x = df_res['resseq'].values

    mean_b = df_res['b_all'].mean()
    ax1.plot(x, df_res['b_mc'], color='#1f4e79', linewidth=1.4, label='Main-chain B-factor')
    ax1.plot(x, df_res['b_sc'], color='#7baaf7', linewidth=1.0, alpha=0.85, label='Side-chain B-factor')
    ax1.axhline(mean_b, color='#555555', linestyle='--', linewidth=1.0, label=f'Mean B ({mean_b:.1f} Å²)')
    ax1.set_ylabel('B-factor (Å²)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Chain {chain_id} Residue-Level Structural Profile', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=False, fontsize=9, ncol=3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    y_h = df_res['hydro_smooth'].values
    ax2.plot(x, y_h, color='#333333', linewidth=1.1)
    ax2.axhline(0, color='#666666', linestyle='--', linewidth=0.9)
    ax2.fill_between(x, y_h, 0, where=(y_h >= 0), color='#d95f02', alpha=0.45, interpolate=True, label='Hydrophobic (>0)')
    ax2.fill_between(x, y_h, 0, where=(y_h < 0), color='#386cb0', alpha=0.45, interpolate=True, label='Hydrophilic (<0)')
    ax2.set_ylabel('Hydrophobicity\n(Kyte-Doolittle)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Residue Number', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', frameon=False, fontsize=9, ncol=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    pocket_res = df_res[df_res['in_pocket']]['resseq'].values
    if len(pocket_res) > 0:
        for r in pocket_res:
            ax1.axvspan(r - 0.5, r + 0.5, color='#ffeb3b', alpha=0.35, lw=0)
            ax2.axvspan(r - 0.5, r + 0.5, color='#ffeb3b', alpha=0.35, lw=0)
        ax1.text(
            0.01, 0.92,
            f'Highlighted: Pocket residues within {dist_cutoff} Å of {selected_lig}',
            transform=ax1.transAxes, fontsize=9, color='#8a6d00', fontweight='bold'
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def build_3d_pdb_string(content, mode='hydro', chain_filter=None):
    """生成用于 3Dmol.js 渲染的 PDB 文本：自动去水；非标准 ATOM 配体自动转为 HETATM；疏水模式替换 B-factor 列"""
    out_lines = []
    b_values = []
    has_ligand = False

    for line in content.splitlines():
        record = line[:6].strip()
        if record == "ATOM":
            resname = line[17:20].strip()
            if resname in WATER_RESNAMES:
                continue
            if resname not in STANDARD_AA:
                has_ligand = True
                out_lines.append("HETATM" + line[6:])
                continue
            chain = line[21].strip() or "A"
            if chain_filter and chain != chain_filter:
                continue
            if len(line) < 66:
                line = line.ljust(66)
            if mode == 'hydro':
                val = get_kd_hydro(resname)
                new_line = line[:60] + f"{val:6.2f}" + line[66:]
                out_lines.append(new_line)
                b_values.append(val)
            else:
                try:
                    b_values.append(float(line[60:66]))
                except ValueError:
                    pass
                out_lines.append(line)
        elif record == "HETATM":
            resname = line[17:20].strip()
            if resname in WATER_RESNAMES:
                continue
            has_ligand = True
            out_lines.append(line)
        elif record in ("TER", "END", "CONECT"):
            out_lines.append(line)

    if mode == 'hydro':
        vmin, vmid, vmax = -4.5, 0.0, 4.5
    else:
        if b_values:
            vmin = float(np.percentile(b_values, 5))
            vmid = float(np.mean(b_values))
            vmax = float(np.percentile(b_values, 95))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmid, vmax = 10.0, 30.0, 60.0

    return "\n".join(out_lines), vmin, vmid, vmax, has_ligand


def render_3dmol_html(pdb_str, mode, style_mode, opacity, vmin, vmid, vmax,
                      has_ligand, pocket_chain=None, pocket_resseqs=None, snap_name="structure_3d"):
    """生成零额外依赖的 3Dmol.js 交互式组件 HTML"""
    if mode == 'hydro':
        c_low, c_mid, c_high = "#386cb0", "#ffffff", "#d95f02"
        legend_title = "Kyte-Doolittle Hydrophobicity"
        lbl_low, lbl_mid, lbl_high = "-4.5 (Hydrophilic)", "0.0 (Neutral)", "+4.5 (Hydrophobic)"
    else:
        c_low, c_mid, c_high = "#1f4e79", "#ffffff", "#d73027"
        legend_title = "B-factor (Å²)"
        lbl_low, lbl_mid, lbl_high = f"{vmin:.1f} (Rigid)", f"{vmid:.1f} (Mean)", f"{vmax:.1f} (Flexible)"

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
      <style>
        body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #ffffff; }
        .toolbar {
          display: flex; align-items: center; justify-content: space-between;
          padding: 8px 12px; background: #f8f9fb; border: 1px solid #e6e9ef; border-bottom: none;
          border-radius: 6px 6px 0 0; font-size: 13px; color: #31333f; flex-wrap: wrap; gap: 8px;
        }
        .btn-group { display: flex; gap: 6px; }
        .btn {
          padding: 5px 11px; border: 1px solid #cfd4dc; border-radius: 5px; background: #ffffff;
          color: #262730; cursor: pointer; font-size: 12.5px; font-weight: 500; transition: all 0.15s;
        }
        .btn:hover { background: #eef2f8; border-color: #9aa5b8; }
        .btn-primary { background: #1f4e79; color: #ffffff; border-color: #1f4e79; }
        .btn-primary:hover { background: #163a5a; }
        #viewer-box {
          width: 100%; height: 520px; position: relative;
          border: 1px solid #e6e9ef; background: #ffffff;
        }
        .legend-bar {
          display: flex; align-items: center; justify-content: center; gap: 14px;
          padding: 10px 14px; background: #f8f9fb; border: 1px solid #e6e9ef; border-top: none;
          border-radius: 0 0 6px 6px; font-size: 12.5px; color: #31333f;
        }
        .grad-strip {
          width: 220px; height: 12px; border-radius: 6px; border: 1px solid #d0d5dd;
          background: linear-gradient(to right, __C_LOW__, __C_MID__, __C_HIGH__);
        }
      </style>
    </head>
    <body>
      <div class="toolbar">
        <span>🖱️ 左键旋转 · 滚轮缩放 · 右键/Shift拖动平移</span>
        <div class="btn-group">
          <button class="btn" onclick="resetView()">🔄 重置全视角</button>
          __FOCUS_BTN__
          <button class="btn btn-primary" onclick="exportSnap()">📷 导出当前 3D 截图 (PNG)</button>
        </div>
      </div>
      <div id="viewer-box"></div>
      <div class="legend-bar">
        <strong>__LEGEND_TITLE__:</strong>
        <span>__LBL_LOW__</span>
        <div class="grad-strip"></div>
        <span>__LBL_MID__</span>
        <span style="color:#aaa;">|</span>
        <span>__LBL_HIGH__</span>
      </div>

      <script>
        const pdbData = __PDB_JSON__;
        const vMin = __VMIN__;
        const vMid = __VMID__;
        const vMax = __VMAX__;
        const styleMode = __STYLE_MODE__;
        const surfOpacity = __OPACITY__;
        const pocketChain = __POCKET_CHAIN__;
        const pocketRes = __POCKET_RES__;
        const snapName = __SNAP_NAME__;

        function hexToRgb(hex) {
          const n = parseInt(hex.slice(1), 16);
          return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
        }
        function rgbToHex(r, g, b) {
          return "#" + [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("");
        }
        const rgbLow = hexToRgb("__C_LOW__");
        const rgbMid = hexToRgb("__C_MID__");
        const rgbHigh = hexToRgb("__C_HIGH__");

        function colorByProp(atom) {
          const val = (atom.b === undefined || isNaN(atom.b)) ? vMid : atom.b;
          let t, c1, c2;
          if (val <= vMid) {
            t = (vMid === vMin) ? 1.0 : Math.max(0, Math.min(1, (val - vMin) / (vMid - vMin)));
            c1 = rgbLow; c2 = rgbMid;
          } else {
            t = (vMax === vMid) ? 1.0 : Math.max(0, Math.min(1, (val - vMid) / (vMax - vMid)));
            c1 = rgbMid; c2 = rgbHigh;
          }
          return rgbToHex(
            c1[0] + (c2[0] - c1[0]) * t,
            c1[1] + (c2[1] - c1[1]) * t,
            c1[2] + (c2[2] - c1[2]) * t
          );
        }

        const element = document.getElementById("viewer-box");
        const viewer = $3Dmol.createViewer(element, { backgroundColor: "white", antialias: true });
        viewer.addModel(pdbData, "pdb");

        if (styleMode === "surface_cartoon" || styleMode === "cartoon") {
          viewer.setStyle({ hetflag: false }, { cartoon: { colorfunc: colorByProp } });
        } else {
          viewer.setStyle({ hetflag: false }, {});
        }

        if (styleMode === "surface_cartoon" || styleMode === "surface") {
          viewer.addSurface(
            $3Dmol.SurfaceType.SES,
            { opacity: surfOpacity, colorfunc: colorByProp },
            { hetflag: false }
          );
        }

        if (pocketRes && pocketRes.length > 0 && pocketChain) {
          viewer.addStyle(
            { hetflag: false, chain: pocketChain, resi: pocketRes },
            { stick: { radius: 0.14, colorscheme: "yellowCarbon" } }
          );
        }

        viewer.setStyle({ hetflag: true }, { stick: { radius: 0.22, colorscheme: "greenCarbon" } });

        viewer.zoomTo();
        viewer.render();

        function resetView() {
          viewer.zoomTo();
          viewer.render();
        }
        function focusLigand() {
          viewer.zoomTo({ hetflag: true }, 600);
          viewer.render();
        }
        function exportSnap() {
          viewer.render();
          const uri = viewer.pngURI();
          const link = document.createElement("a");
          link.href = uri;
          link.download = snapName + ".png";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      </script>
    </body>
    </html>
    """

    focus_btn = '<button class="btn" onclick="focusLigand()">🎯 聚焦配体口袋</button>' if has_ligand else ""
    html = (
        html_template
        .replace("__PDB_JSON__", json.dumps(pdb_str))
        .replace("__VMIN__", f"{vmin:.4f}")
        .replace("__VMID__", f"{vmid:.4f}")
        .replace("__VMAX__", f"{vmax:.4f}")
        .replace("__STYLE_MODE__", json.dumps(style_mode))
        .replace("__OPACITY__", f"{opacity:.2f}")
        .replace("__POCKET_CHAIN__", json.dumps(pocket_chain))
        .replace("__POCKET_RES__", json.dumps(pocket_resseqs or []))
        .replace("__SNAP_NAME__", json.dumps(snap_name))
        .replace("__C_LOW__", c_low)
        .replace("__C_MID__", c_mid)
        .replace("__C_HIGH__", c_high)
        .replace("__LEGEND_TITLE__", legend_title)
        .replace("__LBL_LOW__", lbl_low)
        .replace("__LBL_MID__", lbl_mid)
        .replace("__LBL_HIGH__", lbl_high)
        .replace("__FOCUS_BTN__", focus_btn)
    )
    return html


def run_table1_module():
    st.title("💎 晶体衍射与精修数据分析中心 (Table 1 & 结构评估)")
    st.info("""
    **📖 使用说明：** 请确保上传的文件均来源于同一套处理与精修流程。
    - **1. 数据收集日志 (必传)**：`aimless.log` (CCP4/Xia2/AutoPROC) 或 `scale.log` (HKL2000)。用于提取分辨率、Total/Unique reflections、Rmerge、Rpim、CC1/2、完整度、Wilson B 等指标。
    - **2. Phenix 精修日志 (必传)**：精修完成后的 `.log` 文本。用于提取 Rwork/Rfree、原子数与 B 因子、Clashscore、Rotamer outliers 等指标。
    - **3. 最终 PDB 结构文件 (推荐)**：用于提取权威晶胞参数、拉氏图数据，并解锁**2D 残基特征双轨图与 3D 疏水性/B-factor 交互视图**。
    """)

    example_path = "assets/example_table1.png"
    if os.path.exists(example_path):
        with st.expander("👀 点击查看：处理成功后的标准 Table 1 参考图"):
            st.image(example_path, caption="Standard Table 1 Output", use_container_width=True)

    col_up1, col_up2, col_up3 = st.columns(3)
    with col_up1:
        upload_dcoll = st.file_uploader("1. 数据收集日志 (aimless/scale)", type=['log', 'sca', 'txt', 'out'])
    with col_up2:
        upload_phenix = st.file_uploader("2. Phenix 精修日志 (*refine*.log)", type=['log', 'txt', 'out'])
    with col_up3:
        upload_pdb = st.file_uploader("3. 结构文件 (*.pdb)", type=['pdb'])

    custom_wave = st.text_input("💡 兜底自定义波长 (Å) *(若处理日志中无记录，将使用此值)*", value="0.9792")
    generate_btn = st.button("🚀 一键提取并生成分析结果", type="primary", use_container_width=True)

    if generate_btn:
        dcoll_data, phenix_data, pdb_data = {}, {}, {}
        content_pdb = None
        with st.spinner("正在并发解析特征数据..."):
            if upload_dcoll:
                content_dcoll = upload_dcoll.getvalue().decode('utf-8', errors='ignore')
                if 'AIMLESS' in content_dcoll or '<!--SUMMARY_BEGIN-->' in content_dcoll:
                    dcoll_data = parse_aimless_log(content_dcoll)
                elif 'Summary of reflections intensities and R-factors by shells' in content_dcoll:
                    dcoll_data = parse_scale_log(content_dcoll)
            if upload_pdb:
                content_pdb = upload_pdb.getvalue().decode('utf-8', errors='ignore')
                pdb_data = parse_pdb(content_pdb)
            if upload_phenix:
                content_phenix = upload_phenix.getvalue().decode('utf-8', errors='ignore')
                phenix_data = parse_phenix_log(content_phenix)
                if not upload_pdb and 'REMARK   3' in content_phenix:
                    pdb_data = parse_pdb(content_phenix)
                    st.toast("💡 未检测到独立 PDB，已自动从 Phenix log 尾部提取结构参数。")

        if not dcoll_data and not phenix_data and not pdb_data:
            st.error("❌ 错误：请至少上传一份有效的数据处理或精修日志！")
        else:
            st.session_state['table1_matrix'] = build_table_data(
                dcoll_data, pdb_data, phenix_data, custom_wavelength=custom_wave
            )
            st.session_state['pdb_content'] = content_pdb
            st.session_state['pdb_name'] = os.path.splitext(upload_pdb.name)[0] if upload_pdb else "structure"
            if pdb_data.get('ligand_names'):
                st.toast(f"🎯 已从 PDB 识别到配体/离子: {pdb_data['ligand_names']}")
            st.success("✅ 数据合并解析完成！")

    if 'table1_matrix' in st.session_state:
        tab1, tab2 = st.tabs(["📋 Table 1 统计汇总", "🧬 蛋白残基 B-factor / 疏水性与 3D 口袋分析"])

        with tab1:
            table_matrix = st.session_state['table1_matrix']
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.subheader("📝 文本数据校验")
                st.dataframe(pd.DataFrame(table_matrix, columns=["Parameter", "Value"]), use_container_width=True, height=820)
            with col_res2:
                st.subheader("🖼️ 期刊级图表渲染结果")
                with st.spinner("正在渲染高分辨率图像..."):
                    img_buffer = draw_table_to_buffer(table_matrix)
                    st.image(img_buffer, caption="Table 1: Data collection and refinement statistics", use_container_width=True)
                    st.download_button(
                        "📥 下载高清 PNG (300 DPI)",
                        data=img_buffer,
                        file_name="Table1_Statistics.png",
                        mime="image/png",
                        type="primary",
                        use_container_width=True
                    )

        with tab2:
            pdb_content = st.session_state.get('pdb_content')
            if not pdb_content:
                st.warning("⚠️ 请先在上方上传 `.pdb` 结构文件并点击“一键提取”，即可生成 2D 残基特征图与 3D 疏水性/B-factor 结构视图。")
            else:
                df_atoms, ligands = parse_pdb_for_profile(pdb_content)
                if df_atoms.empty:
                    st.error("❌ 未在 PDB 文件中检测到有效的标准氨基酸 ATOM 记录。")
                else:
                    chains = sorted(df_atoms['chain'].unique().tolist())
                    lig_options = ["不指定配体"] + list(ligands.keys())
                    base_pdb_name = st.session_state.get('pdb_name', 'structure')

                    st.subheader("📈 2D 逐残基 B-factor 与疏水性分布")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        selected_chain = st.selectbox("选择分析蛋白链 (Chain)", chains)
                    with c2:
                        selected_lig = st.selectbox("高亮结合口袋配体 (HETATM)", lig_options)
                    with c3:
                        dist_cutoff = st.slider("口袋接触距离截断 (Å)", 3.0, 6.0, 4.5, 0.1)
                    with c4:
                        window_size = st.slider("疏水性滑动平滑窗口", 1, 9, 5, 2)

                    lig_coords = ligands.get(selected_lig) if selected_lig != "不指定配体" else None
                    df_res = build_residue_profile(
                        df_atoms, selected_chain, lig_coords=lig_coords,
                        dist_cutoff=dist_cutoff, window_size=window_size
                    )

                    pocket_res_list = []
                    if not df_res.empty:
                        pocket_res_list = df_res[df_res['in_pocket']]['resseq'].astype(int).tolist()
                        profile_buf = draw_residue_profile(
                            df_res, selected_chain,
                            selected_lig=(selected_lig if selected_lig != "不指定配体" else None),
                            dist_cutoff=dist_cutoff
                        )
                        st.image(profile_buf, use_container_width=True)

                        col_pdl1, col_pdl2 = st.columns(2)
                        with col_pdl1:
                            st.download_button(
                                "🖼️ 下载残基特征双轨图 (300 DPI PNG)",
                                data=profile_buf,
                                file_name=f"{base_pdb_name}_Chain{selected_chain}_profile.png",
                                mime="image/png",
                                type="primary",
                                use_container_width=True
                            )

                        if lig_coords is not None:
                            df_pocket = df_res[df_res['in_pocket']].copy()
                            st.subheader(f"🎯 结合口袋残基明细 (≤ {dist_cutoff} Å，共 {len(df_pocket)} 个残基)")
                            if not df_pocket.empty:
                                show_cols = ['resseq', 'resname', 'min_dist', 'b_all', 'b_mc', 'b_sc', 'hydro_raw']
                                rename_map = {
                                    'resseq': '残基序号', 'resname': '残基类型', 'min_dist': '距配体最短距离 (Å)',
                                    'b_all': '平均 B (Å²)', 'b_mc': '主链 B (Å²)', 'b_sc': '侧链 B (Å²)',
                                    'hydro_raw': 'Kyte-Doolittle 疏水性'
                                }
                                df_show = df_pocket[show_cols].rename(columns=rename_map)
                                st.dataframe(df_show, use_container_width=True)
                                with col_pdl2:
                                    st.download_button(
                                        "📊 下载结合口袋残基表 (CSV)",
                                        data=df_show.to_csv(index=False).encode('utf-8-sig'),
                                        file_name=f"{base_pdb_name}_Chain{selected_chain}_pocket.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            else:
                                st.info(f"在 Chain {selected_chain} 中未找到距离 {selected_lig} 小于 {dist_cutoff} Å 的接触残基，可尝试切换链或调大截断距离。")
                        else:
                            with col_pdl2:
                                st.download_button(
                                    "📊 下载全链残基特征数据 (CSV)",
                                    data=df_res.to_csv(index=False).encode('utf-8-sig'),
                                    file_name=f"{base_pdb_name}_Chain{selected_chain}_residues.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                    # --- 3D 结构交互可视化 ---
                    st.markdown("---")
                    st.subheader("🌐 3D 结构疏水性 / B-factor 表面可视化（已自动去除水分子）")
                    v1, v2, v3, v4 = st.columns(4)
                    with v1:
                        color_mode_label = st.selectbox(
                            "3D 着色映射模式",
                            ["疏水性表面 (Kyte-Doolittle)", "B-factor 柔性分布"]
                        )
                    with v2:
                        style_label = st.selectbox(
                            "3D 展示样式",
                            ["分子表面 + 飘带", "纯分子表面 (Surface)", "纯飘带 (Cartoon)"]
                        )
                    with v3:
                        scope_label = st.selectbox(
                            "显示蛋白链范围",
                            [f"仅当前链 (Chain {selected_chain})", "全部蛋白链"]
                        )
                    with v4:
                        surf_opacity = st.slider("分子表面透明度", 0.20, 1.00, 0.85, 0.05)

                    show_pocket_sticks = False
                    if pocket_res_list:
                        show_pocket_sticks = st.checkbox(
                            f"高亮显示与 {selected_lig} 接触的 {len(pocket_res_list)} 个口袋残基侧链 (黄色 Sticks)",
                            value=True
                        )

                    mode_key = 'hydro' if "疏水性" in color_mode_label else 'bfactor'
                    style_map = {
                        "分子表面 + 飘带": "surface_cartoon",
                        "纯分子表面 (Surface)": "surface",
                        "纯飘带 (Cartoon)": "cartoon"
                    }
                    chain_filter = selected_chain if "仅当前链" in scope_label else None

                    pdb_3d_str, vmin_3d, vmid_3d, vmax_3d, has_lig_3d = build_3d_pdb_string(
                        pdb_content, mode=mode_key, chain_filter=chain_filter
                    )
                    snap_suffix = "hydrophobicity" if mode_key == 'hydro' else "bfactor"
                    html_3d = render_3dmol_html(
                        pdb_3d_str,
                        mode=mode_key,
                        style_mode=style_map[style_label],
                        opacity=surf_opacity,
                        vmin=vmin_3d,
                        vmid=vmid_3d,
                        vmax=vmax_3d,
                        has_ligand=has_lig_3d,
                        pocket_chain=selected_chain if show_pocket_sticks else None,
                        pocket_resseqs=pocket_res_list if show_pocket_sticks else [],
                        snap_name=f"{base_pdb_name}_{snap_suffix}_3D"
                    )
                    components.html(html_3d, height=620, scrolling=False)


# =====================================================================
#                     模块 2: CBF 衍射图像在线分析工具
# =====================================================================
def parse_cbf_header(cbf_obj):
    params = {
        'wavelength': 0.97918, 'distance': 200.0, 'pixel_size': 0.075,
        'beam_x': cbf_obj.data.shape[1] / 2, 'beam_y': cbf_obj.data.shape[0] / 2
    }
    header_text = cbf_obj.header.get('_array_data.header_contents', '')
    float_re = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'

    match_dist = re.search(rf'Detector_distance\s+{float_re}\s*([a-zA-Z]*)', header_text, re.IGNORECASE)
    if match_dist:
        dist_val, unit = float(match_dist.group(1)), match_dist.group(2).strip().lower()
        if unit == 'mm':
            params['distance'] = dist_val
        elif unit == 'm':
            params['distance'] = dist_val * 1000
        else:
            params['distance'] = dist_val * 1000 if dist_val < 5.0 else dist_val

    match_wl = re.search(rf'Wavelength\s+{float_re}', header_text, re.IGNORECASE)
    if match_wl:
        wl_val = float(match_wl.group(1))
        if 0.1 < wl_val < 5.0:
            params['wavelength'] = wl_val

    match_px = re.search(rf'Pixel_size\s+{float_re}\s*([a-zA-Z]*)', header_text, re.IGNORECASE)
    if match_px:
        px_val, unit = float(match_px.group(1)), match_px.group(2).strip().lower()
        if unit == 'm' or px_val < 0.01:
            params['pixel_size'] = px_val * 1000
        else:
            params['pixel_size'] = px_val

    match_beam = re.search(rf'Beam_xy \(pixels\)\s+{float_re},\s+{float_re}', header_text, re.IGNORECASE)
    if match_beam:
        params['beam_x'], params['beam_y'] = float(match_beam.group(1)), float(match_beam.group(2))
    else:
        match_bx = re.search(rf'Beam_center_x\s+{float_re}', header_text, re.IGNORECASE)
        match_by = re.search(rf'Beam_center_y\s+{float_re}', header_text, re.IGNORECASE)
        if match_bx and match_by:
            params['beam_x'], params['beam_y'] = float(match_bx.group(1)), float(match_by.group(1))
    return params


def filter_ice_rings(peaks, beam_x, beam_y, pixel_size, distance, wavelength, tol=0.035):
    ice_d_spacings = [3.897, 3.669, 3.441, 2.671, 2.249, 2.072, 1.948, 1.883, 1.718]
    valid_peaks = []
    for p in peaks:
        y, x = p[0], p[1]
        r_mm = np.sqrt((x - beam_x) ** 2 + (y - beam_y) ** 2) * pixel_size
        theta = 0.5 * np.arctan(r_mm / distance)
        if theta > 0:
            d = wavelength / (2 * np.sin(theta))
            if not any(abs(d - ice_d) < tol for ice_d in ice_d_spacings):
                valid_peaks.append(p)
    return np.array(valid_peaks)


def run_cbf_module():
    st.title("🔬 CBF 衍射图像在线分析工具")
    st.info("""
    **📖 使用说明：**
    上传结构生物学单晶衍射生成的 `.cbf` 原始图像，工具将自动解析 Header 中的波长、Detector 距离和 Beam Center。
    开启 **Peak Search** 可自动寻找衍射斑点，并内置物理冰环过滤算法剔除水冰干扰。
    """)

    example_path = "assets/example_cbf.png"
    if os.path.exists(example_path):
        with st.expander("👀 点击查看：处理成功后的标准 CBF 分析图"):
            st.image(example_path, caption="Standard CBF Peak Search Output", use_container_width=True)

    col_up, _ = st.columns([1, 2])
    with col_up:
        uploaded_file = st.file_uploader("请选择一张 .cbf 衍射图", type=["cbf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cbf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            cbf_img = fabio.open(tmp_path)
            raw_data = cbf_img.data
            header_params = parse_cbf_header(cbf_img)
        finally:
            os.remove(tmp_path)

        col_ctrl, col_plot = st.columns([1, 3])
        with col_ctrl:
            st.header("⚙️ 图像显示调节")
            cmap_choice = st.selectbox("配色方案", ["gray_r", "gray", "viridis"])
            vmax_percentile = st.slider("对比度调节", 96.00, 99.99, 99.99, 0.01)

            display_data = raw_data
            vmin, vmax = np.min(display_data), np.percentile(display_data, vmax_percentile)
            wavelength = float(header_params['wavelength'])
            distance = float(header_params['distance'])
            pixel_size = float(header_params['pixel_size'])
            beam_x, beam_y = float(header_params['beam_x']), float(header_params['beam_y'])

            img_h, img_w = raw_data.shape
            max_radius_mm = min(beam_x, img_w - beam_x, beam_y, img_h - beam_y) * pixel_size
            edge_theta = 0.5 * np.arctan(max_radius_mm / distance)
            edge_d_rounded = round((wavelength / (2 * np.sin(edge_theta))) if edge_theta > 0 else 1.0, 2)

            base_rings = [5.0, 3.0, 2.0]
            display_rings_float = [r for r in base_rings if r >= edge_d_rounded]
            if not display_rings_float or (display_rings_float[-1] - edge_d_rounded) > 0.3:
                display_rings_float.append(edge_d_rounded)

            st.header("⭕ 分辨率环设置")
            show_rings = st.checkbox("开启分辨率环", value=True)
            if show_rings:
                ring_color = st.color_picker("🎨 分辨率环颜色", "#00008B")
                st.info(f"探测器边缘极限分辨率约为 **{edge_d_rounded} Å**")
                res_input = st.text_input("目标分辨率环 (Å)", ", ".join([str(r) for r in display_rings_float]))

            st.header("✨ 衍射点自动标注")
            show_peaks = st.checkbox("开启 Peak Search", value=False)
            if show_peaks:
                auto_filter_ice = st.checkbox("物理消除冰环假阳性", value=True)
                min_intensity = st.slider("净信号斑点强度阈值", 5, 200, 20)
                bg_sigma = st.slider("背景模糊半径", 5, 30, 15)
                min_dist = st.slider("斑点间最小像素距离", 1, 30, 4)

        with col_plot:
            st.subheader("衍射图可视化")
            fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
            ax.imshow(display_data, cmap=cmap_choice, vmin=vmin, vmax=vmax, origin='lower')

            if show_rings:
                try:
                    for d in [float(r.strip()) for r in res_input.split(",")]:
                        radius_px = (distance * np.tan(2 * np.arcsin(wavelength / (2 * d)))) / pixel_size
                        ax.add_patch(Circle((beam_x, beam_y), radius_px, color=ring_color, fill=False, linestyle='--', linewidth=1, alpha=0.7))
                        ax.text(beam_x, beam_y + radius_px + 10, f"{d} Å", color=ring_color, ha='center', va='bottom', fontsize=9)
                    ax.plot(beam_x, beam_y, marker='+', color=ring_color, markersize=10, label='Beam Center')
                except Exception:
                    st.error("分辨率输入格式有误。")

            peaks_data = None
            if show_peaks:
                with st.spinner("正在检索衍射斑点并过滤干扰..."):
                    signal_data = np.clip(raw_data.astype(float) - gaussian_filter(raw_data.astype(float), sigma=bg_sigma), 0, None)
                    peaks = peak_local_max(signal_data, min_distance=min_dist, threshold_abs=min_intensity)
                    if len(peaks) > 0:
                        if auto_filter_ice:
                            peaks = filter_ice_rings(peaks, beam_x, beam_y, pixel_size, distance, wavelength)
                        if len(peaks) > 0:
                            peak_x, peak_y = peaks[:, 1], peaks[:, 0]
                            ax.scatter(peak_x, peak_y, edgecolors='red', facecolors='none', s=20, linewidths=0.6, alpha=0.8)
                            st.success(f"✅ 成功找到 {len(peaks)} 个有效衍射斑点！")
                            peaks_data = pd.DataFrame({"X (px)": peak_x, "Y (px)": peak_y, "Intensity": raw_data[peak_y, peak_x]})
                        else:
                            st.warning("初步找到的斑点均落在冰环位置并已被剔除。")
                    else:
                        st.warning("未找到符合要求的衍射点，请尝试调低阈值。")

            ax.set_aspect('equal')
            ax.axis('off')
            fig.tight_layout()
            st.pyplot(fig)

            base_filename = os.path.splitext(uploaded_file.name)[0]
            col_dl1, col_dl2 = st.columns(2)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            with col_dl1:
                st.download_button(
                    "🖼️ 下载当前视图图像 (PNG)",
                    data=buf,
                    file_name=f"{base_filename}_view.png",
                    mime="image/png"
                )
            with col_dl2:
                if show_peaks and peaks_data is not None:
                    st.download_button(
                        "📊 下载斑点坐标 (CSV)",
                        data=peaks_data.to_csv(index=False).encode('utf-8'),
                        file_name=f"{base_filename}_peaks.csv",
                        mime="text/csv"
                    )


# =====================================================================
#                          主程序侧边栏路由
# =====================================================================
def main():
    st.sidebar.title("🧬 数据处理工具箱")
    st.sidebar.markdown("请在下方选择所需功能模块：")

    app_mode = st.sidebar.radio("模块导航", [
        "📊 模块 1：晶体衍射数据汇总表",
        "🔬 模块 2：CBF 衍射图像查看器"
    ])

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 结构生物学自动化中心")

    if app_mode == "📊 模块 1：晶体衍射数据汇总表":
        run_table1_module()
    elif app_mode == "🔬 模块 2：CBF 衍射图像查看器":
        run_cbf_module()


if __name__ == "__main__":
    main()
