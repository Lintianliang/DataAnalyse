import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.feature import peak_local_max
import fabio
import re
import io
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="结构生物学数据分析中心", layout="wide", page_icon="🧬")


# =====================================================================
#                     模块 1: Table 1 核心引擎与页面
# =====================================================================
def parse_aimless_log(content):
    data = {}
    summary_match = re.search(r'<!--SUMMARY_BEGIN-->(.*?)<!--SUMMARY_END-->', content, re.DOTALL)
    summary = summary_match.group(1) if summary_match else content
    def get_vals(pattern, source=summary):
        match = re.search(pattern, source)
        return match.groups() if match else (None, None, None)
    m_cell = re.search(r'Average unit cell:\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', content)
    if m_cell:
        data['a'], data['b'], data['c'] = m_cell.group(1), m_cell.group(2), m_cell.group(3)
        data['alpha'], data['beta'], data['gamma'] = m_cell.group(4), m_cell.group(5), m_cell.group(6)
    m_sg = re.search(r'Space group[^\n]*:\s*(.*)', content)
    if m_sg: data['space_group'] = m_sg.group(1).strip()
    m_wave = re.search(r'Wavelength:\s+([\d\.]+)', content)
    if m_wave: data['wavelength'] = m_wave.group(1)
    m_low = get_vals(r'Low resolution limit\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_low[0]: data['low_res'] = m_low[0]
    m_high = get_vals(r'High resolution limit\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_high[0]: data['high_res'], data['high_res_outer'] = m_high[0], m_high[2]
    m_rmerge = get_vals(r'Rmerge\s*\(all I\+\s*(?:and|&)\s*I-\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_rmerge[0]: data['rmerge_overall'], data['rmerge_outer'] = m_rmerge[0], m_rmerge[2]
    m_cc12 = get_vals(r'Mn\(I\) half-set correlation CC\(1/2\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if not m_cc12[0]: m_cc12 = get_vals(r'CC\(1/2\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_cc12[0]: data['cc12_overall'], data['cc12_outer'] = m_cc12[0], m_cc12[2]
    m_isigi = get_vals(r'Mean\(\(I\)/sd\(I\)\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if not m_isigi[0]: m_isigi = get_vals(r'Mean I/sigI\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_isigi[0]: data['isigi_overall'], data['isigi_outer'] = m_isigi[0], m_isigi[2]
    m_comp = get_vals(r'Completeness\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_comp[0]: data['comp_overall'], data['comp_outer'] = m_comp[0], m_comp[2]
    m_mult = get_vals(r'Multiplicity\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
    if m_mult[0]: data['mult_overall'], data['mult_outer'] = m_mult[0], m_mult[2]
    return data

def parse_scale_log(content):
    data = {}
    m_cell = re.findall(r'Cell,\s*a\s+([\d\.]+)\s+b\s+([\d\.]+)\s+c\s+([\d\.]+)\s+alpha\s+([\d\.]+)\s+beta\s+([\d\.]+)\s+gamma\s+([\d\.]+)', content)
    if m_cell:
        data['a'], data['b'], data['c'] = m_cell[-1][0], m_cell[-1][1], m_cell[-1][2]
        data['alpha'], data['beta'], data['gamma'] = m_cell[-1][3], m_cell[-1][4], m_cell[-1][5]
    m_sg = re.search(r'Space group\s+(.*?)\s+number', content, re.IGNORECASE)
    if m_sg: data['space_group'] = m_sg.group(1).strip()
    lines = content.split('\n')
    in_r_table, r_table = False, []
    for line in lines:
        if "Summary of reflections intensities and R-factors by shells" in line: in_r_table, r_table = True, []
        elif in_r_table:
            if not line.strip() and len(r_table) > 5: in_r_table = False
            else: r_table.append(line.strip())
    if r_table:
        data_lines = [l.split() for l in r_table if l and l[0].replace('.','',1).isdigit()]
        if data_lines:
            outer = data_lines[-1]
            data['low_res'], data['high_res'], data['high_res_outer'] = data_lines[0][0], outer[1], outer[1]
            data['rmerge_outer'] = outer[6]
            try: data['isigi_outer'] = f"{float(outer[2])/float(outer[3]):.1f}"
            except: pass
            if len(outer) > 10: data['cc12_outer'] = outer[10]
        for l in r_table:
            if l.startswith("All reflections") or l.startswith("All hkl"):
                parts = l.split()
                data['rmerge_overall'] = parts[6]
                try: data['isigi_overall'] = f"{float(parts[2])/float(parts[3]):.1f}"
                except: pass
                if len(parts) > 10: data['cc12_overall'] = parts[10]
    in_comp, comp_table = False, []
    for line in lines:
        if "Summary of observation redundancies:" in line: in_comp, comp_table = True, []
        elif in_comp:
            if not line.strip() and len(comp_table) > 5: in_comp = False
            else: comp_table.append(line.strip())
    if comp_table:
        data_lines = [l.split() for l in comp_table if l and l[0].replace('.','',1).isdigit()]
        if data_lines: data['comp_outer'] = data_lines[-1][-1]
        for l in comp_table:
            if l.startswith("All hkl"): data['comp_overall'] = l.split()[-1]
    in_red, red_table = False, []
    for line in lines:
        if "Average Redundancy Per Shell" in line: in_red, red_table = True, []
        elif in_red:
            if not line.strip() and len(red_table) > 3: in_red = False
            else: red_table.append(line.strip())
    if red_table:
        data_lines = [l.split() for l in red_table if l and l[0].replace('.','',1).isdigit()]
        if data_lines: data['mult_outer'] = data_lines[-1][-1]
        for l in red_table:
            if l.startswith("All hkl"): data['mult_overall'] = l.split()[-1]
    return data

def parse_pdb(content):
    data = {}
    if not content: return data
    for line in content.splitlines():
        if line.startswith("CRYST1"):
            data['space_group'] = line[55:66].strip()
            data['a'], data['b'], data['c'] = line[6:15].strip(), line[15:24].strip(), line[24:33].strip()
            data['alpha'], data['beta'], data['gamma'] = line[33:40].strip(), line[40:47].strip(), line[47:54].strip()
        if line.startswith("REMARK   3"):
            m_bond = re.search(r'BOND\s+:\s+([\d\.]+)', line)
            if m_bond: data['rmsd_bonds'] = m_bond.group(1)
            m_angle = re.search(r'ANGLE\s+:\s+([\d\.]+)', line)
            if m_angle: data['rmsd_angles'] = m_angle.group(1)
            if 'NUMBER OF REFLECTIONS' in line and 'NON-ANOMALOUS' not in line:
                m_tot = re.search(r':\s*(\d+)', line)
                if m_tot: data['refl_total'] = m_tot.group(1)
            if 'FREE R VALUE TEST SET COUNT' in line or 'NUMBER OF REFLECTIONS IN TEST SET' in line:
                m_free = re.search(r':\s*(\d+)', line)
                if m_free: data['refl_free'] = m_free.group(1)
            if 'FAVORED' in line and '%' in line:
                m_fav = re.search(r':\s*([\d\.]+)', line)
                if m_fav: data['rama_favored'] = m_fav.group(1)
            if 'ALLOWED' in line and '%' in line:
                m_all = re.search(r':\s*([\d\.]+)', line)
                if m_all: data['rama_allowed'] = m_all.group(1)
            if 'OUTLIERS' in line and '%' in line:
                m_out = re.search(r':\s*([\d\.]+)', line)
                if m_out: data['rama_outlier'] = m_out.group(1)
        if line.startswith("ATOM") or line.startswith("HETATM"): break
    if 'refl_total' in data and 'refl_free' in data:
        data['refl_work'] = str(int(data['refl_total']) - int(data['refl_free']))
    return data

def parse_phenix_log(content):
    data = {}
    m_r = re.search(r'Final R-work\s*=\s*([\d\.]+),\s*R-free\s*=\s*([\d\.]+)', content)
    if m_r: data['r_work'], data['r_free'] = m_r.group(1), m_r.group(2)
    m_res = re.findall(r'resolution:\s*([\d\.]+)\s*-\s*([\d\.]+)\s*[AÅ],\s*n_refl\.=(\d+)', content)
    if m_res:
        r1, r2 = float(m_res[-1][0]), float(m_res[-1][1])
        data['refine_res_high'], data['refine_res_low'] = f"{min(r1, r2):.2f}", f"{max(r1, r2):.2f}"
    m_prot = re.findall(r'Protein:\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)\s+[^\s]+\s+(\d+)', content)
    if m_prot: data['b_protein'], data['protein_atoms'] = m_prot[-1][0], m_prot[-1][1]
    m_wat = re.findall(r'Water:\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)\s+[^\s]+\s+(\d+)', content)
    if m_wat: data['b_water'], data['water_atoms'] = m_wat[-1][0], m_wat[-1][1]
    return data

def build_table_data(dcoll, pdb, phenix, custom_wavelength="0.9792"):
    def br(v1, v2): return f"{v1} ({v2})" if v1 and v2 else v1 or "N/A"
    r_work = f"{float(phenix['r_work'])*100:.2f}" if 'r_work' in phenix else 'N/A'
    r_free = f"{float(phenix['r_free'])*100:.2f}" if 'r_free' in phenix else 'N/A'
    rama = "N/A"
    if 'rama_favored' in pdb: rama = f"{pdb.get('rama_favored', '0')}/{pdb.get('rama_allowed', '0')}/{pdb.get('rama_outlier', '0')}"
    refl = f"{pdb['refl_work']}/{pdb['refl_free']}" if 'refl_work' in pdb and 'refl_free' in pdb else "N/A"
    a = pdb.get('a') or dcoll.get('a') or 'N/A'
    b = pdb.get('b') or dcoll.get('b') or 'N/A'
    c = pdb.get('c') or dcoll.get('c') or 'N/A'
    alpha = pdb.get('alpha') or dcoll.get('alpha') or 'N/A'
    beta = pdb.get('beta') or dcoll.get('beta') or 'N/A'
    gamma = pdb.get('gamma') or dcoll.get('gamma') or 'N/A'
    sg = pdb.get('space_group') or dcoll.get('space_group') or 'N/A'
    final_wavelength = dcoll.get('wavelength') or custom_wavelength
    return [
        ["Data Collection", ""], ["Wavelength (Å)", final_wavelength], ["Space group", sg], ["Cell parameters", ""],
        ["a, b, c (Å)", f"{a}, {b}, {c}"], ["α, β, γ (°)", f"{alpha}, {beta}, {gamma}"],
        ["Resolution (Å)", f"{dcoll.get('low_res', 'N/A')}–{dcoll.get('high_res', 'N/A')}\n({dcoll.get('high_res', 'N/A')}–{dcoll.get('high_res_outer', 'N/A')})"],
        ["Rmerge (%)", br(dcoll.get('rmerge_overall'), dcoll.get('rmerge_outer'))],
        ["CC1/2", br(dcoll.get('cc12_overall'), dcoll.get('cc12_outer'))], ["I/σI", br(dcoll.get('isigi_overall'), dcoll.get('isigi_outer'))],
        ["Completeness (%)", br(dcoll.get('comp_overall'), dcoll.get('comp_outer'))], ["Redundancy", br(dcoll.get('mult_overall'), dcoll.get('mult_outer'))],
        ["Refinement", ""], ["No. reflections used/free", refl],
        ["Resolution (Å)", f"{phenix.get('refine_res_low', 'N/A')}–{phenix.get('refine_res_high', 'N/A')}"],
        ["Rwork / Rfree (%)", f"{r_work}/{r_free}"], ["R.m.s. deviations", ""],
        ["Bonds lengths (Å)", pdb.get('rmsd_bonds', 'N/A')], ["Bond angles (°)", pdb.get('rmsd_angles', 'N/A')],
        ["B-factors (Å²)", ""], ["Protein", phenix.get('b_protein', 'N/A')], ["Water", phenix.get('b_water', 'N/A')],
        ["No. atoms", ""], ["Protein", phenix.get('protein_atoms', 'N/A')], ["Water", phenix.get('water_atoms', 'N/A')],
        ["Ramachandran plot", ""], ["Favored/allowed/outlier (%)", rama]
    ]

@st.cache_data
def draw_table_to_buffer(table_content):
    df = pd.DataFrame(table_content, columns=["Parameter", "Value"])
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.35 + 1))
    ax.axis('off'); ax.axis('tight')
    table = ax.table(cellText=df.values, loc='center', cellLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white'); cell.set_facecolor('white')
        text = cell.get_text().get_text()
        if text in ["Data Collection", "Refinement"]:
            cell.set_text_props(weight='bold')
            cell.set_edgecolor('black'); cell.visible_edges = 'TB'; cell.set_linewidth(1.5)
        if col == 0 and row > 0 and text not in ["Data Collection", "Wavelength (Å)", "Space group", "Cell parameters", "Resolution (Å)", "Rmerge (%)", "CC1/2", "I/σI", "Completeness (%)", "Redundancy", "Refinement", "No. reflections used/free", "Rwork / Rfree (%)", "R.m.s. deviations", "B-factors (Å²)", "No. atoms", "Ramachandran plot"]:
            cell.get_text().set_text("    " + text)
        if col == 0: cell.PAD = 0.05
    for col in range(2):
        table.get_celld()[(0, col)].set_edgecolor('black'); table.get_celld()[(0, col)].visible_edges = 'T'; table.get_celld()[(0, col)].set_linewidth(1.5)
        table.get_celld()[(len(df)-1, col)].set_edgecolor('black'); table.get_celld()[(len(df)-1, col)].visible_edges = 'B'; table.get_celld()[(len(df)-1, col)].set_linewidth(1.5)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def run_table1_module():
    st.title("💎 晶体衍射与精修数据汇总表 (Table 1)")
    st.info("""
    **📖 使用说明：** 为了生成准确无误的 Table 1，请**确保上传的文件均来源于同一套处理与精修流程**。
    - **1. 数据收集日志 (必传)**：`aimless.log` (CCP4/Xia2/AutoPROC) 或 `scale.log` (HKL2000)。用于提取分辨率、Rmerge、CC1/2、完整度等指标。
    - **2. Phenix 精修日志 (必传)**：精修完成后的 `.log` 文本。用于提取 Rwork/Rfree、原子数量、B 因子等。
    - **3. 最终 PDB 结构文件 (推荐)**：即提交至数据库前的 `.pdb` 文件。包含最权威的晶胞参数及拉氏图数据。
    """)
    
    example_path = "assets/example_table1.png"
    if os.path.exists(example_path):
        with st.expander("👀 点击查看：处理成功后的标准 Table 1 参考图"):
            st.image(example_path, caption="Standard Table 1 Output", use_container_width=True)
            
    col_up1, col_up2, col_up3 = st.columns(3)
    with col_up1: upload_dcoll = st.file_uploader("1. 数据收集日志 (aimless/scale)", type=['log', 'sca', 'txt', 'out'])
    with col_up2: upload_phenix = st.file_uploader("2. Phenix 精修日志 (*refine*.log)", type=['log', 'txt', 'out'])
    with col_up3: upload_pdb = st.file_uploader("3. 结构文件 (*.pdb)", type=['pdb'])
    
    custom_wave = st.text_input("💡 兜底自定义波长 (Å) *(若处理日志中无记录，将使用此值)*", value="0.9792")
    generate_btn = st.button("🚀 一键提取并生成 Table 1", type="primary", use_container_width=True)

    if generate_btn:
        dcoll_data, phenix_data, pdb_data = {}, {}, {}
        with st.spinner("正在并发解析各种特征数据..."):
            if upload_dcoll:
                content_dcoll = upload_dcoll.getvalue().decode('utf-8', errors='ignore')
                if 'AIMLESS' in content_dcoll or '<!--SUMMARY_BEGIN-->' in content_dcoll: dcoll_data = parse_aimless_log(content_dcoll)
                elif 'Summary of reflections intensities and R-factors by shells' in content_dcoll: dcoll_data = parse_scale_log(content_dcoll)
            if upload_pdb:
                content_pdb = upload_pdb.getvalue().decode('utf-8', errors='ignore')
                pdb_data = parse_pdb(content_pdb)
            if upload_phenix:
                content_phenix = upload_phenix.getvalue().decode('utf-8', errors='ignore')
                phenix_data = parse_phenix_log(content_phenix)
                if not upload_pdb and 'REMARK   3' in content_phenix:
                    pdb_data = parse_pdb(content_phenix)

        if not dcoll_data and not phenix_data and not pdb_data: st.error("❌ 错误：请至少上传一份有效的数据处理或精修日志！")
        else:
            st.success("✅ 数据合并解析完成！")
            table_matrix = build_table_data(dcoll_data, pdb_data, phenix_data, custom_wavelength=custom_wave)
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.subheader("📝 文本数据校验")
                st.dataframe(pd.DataFrame(table_matrix, columns=["Parameter", "Value"]), use_container_width=True, height=750)
            with col_res2:
                st.subheader("🖼️ 期刊级图表渲染结果")
                with st.spinner("正在渲染高分辨率图像..."):
                    img_buffer = draw_table_to_buffer(table_matrix)
                    st.image(img_buffer, caption="Table 1: Data collection and refinement statistics", use_container_width=True)
                    st.download_button("📥 下载高清 PNG (300 DPI)", data=img_buffer, file_name="Table1_Statistics.png", mime="image/png", type="primary", use_container_width=True)


# =====================================================================
#                     模块 2: CBF 衍射图像在线分析工具
# =====================================================================
def parse_cbf_header(cbf_obj):
    params = {'wavelength': 0.97918, 'distance': 200.0, 'pixel_size': 0.075, 'beam_x': cbf_obj.data.shape[1]/2, 'beam_y': cbf_obj.data.shape[0]/2}
    header_text = cbf_obj.header.get('_array_data.header_contents', '')
    float_re = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    match_dist = re.search(rf'Detector_distance\s+{float_re}\s*([a-zA-Z]*)', header_text, re.IGNORECASE)
    if match_dist:
        dist_val, unit = float(match_dist.group(1)), match_dist.group(2).strip().lower()
        if unit == 'mm': params['distance'] = dist_val
        elif unit == 'm': params['distance'] = dist_val * 1000
        else: params['distance'] = dist_val * 1000 if dist_val < 5.0 else dist_val
    match_wl = re.search(rf'Wavelength\s+{float_re}', header_text, re.IGNORECASE)
    if match_wl: 
        wl_val = float(match_wl.group(1))
        if 0.1 < wl_val < 5.0: params['wavelength'] = wl_val
    match_px = re.search(rf'Pixel_size\s+{float_re}\s*([a-zA-Z]*)', header_text, re.IGNORECASE)
    if match_px:
        px_val, unit = float(match_px.group(1)), match_px.group(2).strip().lower()
        if unit == 'm' or px_val < 0.01: params['pixel_size'] = px_val * 1000
        else: params['pixel_size'] = px_val
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
        r_mm = np.sqrt((x - beam_x)**2 + (y - beam_y)**2) * pixel_size
        theta = 0.5 * np.arctan(r_mm / distance)
        if theta > 0:
            d = wavelength / (2 * np.sin(theta))
            if not any(abs(d - ice_d) < tol for ice_d in ice_d_spacings): valid_peaks.append(p)
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
    with col_up: uploaded_file = st.file_uploader("请选择一张 .cbf 衍射图", type=["cbf"])

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
            wavelength, distance, pixel_size = float(header_params['wavelength']), float(header_params['distance']), float(header_params['pixel_size'])
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
                except: st.error("分辨率输入格式有误。")

            peaks_data = None
            if show_peaks:
                with st.spinner("正在检索衍射斑点并过滤干扰..."):
                    signal_data = np.clip(raw_data.astype(float) - gaussian_filter(raw_data.astype(float), sigma=bg_sigma), 0, None)
                    peaks = peak_local_max(signal_data, min_distance=min_dist, threshold_abs=min_intensity)
                    if len(peaks) > 0:
                        if auto_filter_ice: peaks = filter_ice_rings(peaks, beam_x, beam_y, pixel_size, distance, wavelength)
                        if len(peaks) > 0:
                            peak_x, peak_y = peaks[:, 1], peaks[:, 0]
                            ax.scatter(peak_x, peak_y, edgecolors='red', facecolors='none', s=20, linewidths=0.6, alpha=0.8)
                            st.success(f"✅ 成功找到 {len(peaks)} 个有效衍射斑点！")
                            peaks_data = pd.DataFrame({"X (px)": peak_x, "Y (px)": peak_y, "Intensity": raw_data[peak_y, peak_x]})
                        else: st.warning("初步找到的斑点均落在冰环位置并已被剔除。")
                    else: st.warning("未找到符合要求的衍射点，请尝试调低阈值。")

            ax.set_aspect('equal'); ax.axis('off'); fig.tight_layout()
            st.pyplot(fig)
            
            base_filename = os.path.splitext(uploaded_file.name)[0]
            col_dl1, col_dl2 = st.columns(2)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            
            with col_dl1: st.download_button("🖼️ 下载当前视图图像 (PNG)", data=buf, file_name=f"{base_filename}_view.png", mime="image/png")
            with col_dl2:
                if show_peaks and peaks_data is not None:
                    st.download_button("📊 下载斑点坐标 (CSV)", data=peaks_data.to_csv(index=False).encode('utf-8'), file_name=f"{base_filename}_peaks.csv", mime="text/csv")


# =====================================================================
#                     模块 3: DSF 熔解曲线分析
# =====================================================================
def get_dynamic_window(data_length):
    window = int(data_length * 0.06)
    if window < 5: window = 5
    if window > 11: window = 11 
    return window if window % 2 != 0 else window + 1

def calculate_auto_qc_params(clean_df, well_map, excludes=[]):
    raw_bgs, raw_amps = [], []
    for w_key, w_col in well_map.items():
        if w_key in excludes: continue 
        f_vals = pd.to_numeric(clean_df[w_col], errors='coerce').dropna().values
        if len(f_vals) >= 10:
            raw_bgs.append(np.mean(f_vals[:5]))
            raw_amps.append(np.max(f_vals) - np.min(f_vals))
    if not raw_amps: return 400000, 30000
    max_amp = np.max(raw_amps)
    active_bgs = [bg for bg, amp in zip(raw_bgs, raw_amps) if amp > max_amp * 0.1]
    active_amps = [amp for amp in raw_amps if amp > max_amp * 0.1]
    if active_bgs and active_amps: return np.median(active_bgs) * 5, np.median(active_amps) * 0.2 
    return np.median(raw_bgs) * 5, np.median(raw_amps) * 0.2 

def check_curve_quality(fluor_series, max_bg, min_amp):
    f_vals = pd.to_numeric(fluor_series, errors='coerce').dropna().values
    if len(f_vals) < 10: return False, "数据过短"
    init_bg = np.mean(f_vals[:5])
    amplitude = np.max(f_vals) - np.min(f_vals)
    if init_bg > max_bg: return False, f"高自发荧光 (BG:{init_bg:.0f}>Max:{max_bg:.0f})"
    if amplitude < min_amp: return False, f"曲线水平/淬灭 (Amp:{amplitude:.0f}<Min:{min_amp:.0f})"
    return True, "Pass"

def get_curve_arrays(temp_series, fluor_series, use_advanced_fitting=False):
    try:
        f_vals = pd.to_numeric(fluor_series, errors='coerce').astype(float).values
        t_vals = pd.to_numeric(temp_series, errors='coerce').astype(float).values
        mask = ~np.isnan(f_vals) & ~np.isnan(t_vals)
        f_vals, t_vals = f_vals[mask], t_vals[mask]
        if len(f_vals) < 10: return None, None, None, None, np.nan
        
        if use_advanced_fitting:
            window_f = get_dynamic_window(len(f_vals))
            window_f = int(window_f * 1.5)  
            window_f = window_f if window_f % 2 != 0 else window_f + 1 
            smoothed_f = savgol_filter(f_vals, window_length=window_f, polyorder=3)
            raw_derivative = np.gradient(smoothed_f, t_vals)
            window_d = int(len(f_vals) * 0.15) 
            window_d = max(7, window_d)
            window_d = window_d if window_d % 2 != 0 else window_d + 1
            derivative = savgol_filter(raw_derivative, window_length=window_d, polyorder=2)
            
            start_idx, end_idx = int(len(derivative) * 0.1), int(len(derivative) * 0.9)
            local_max_idx = np.argmax(derivative[start_idx:end_idx])
            max_idx = local_max_idx + start_idx
            
            max_val = derivative[max_idx]
            threshold = max_val * 0.80
            peak_indices = np.where((derivative[start_idx:end_idx] > threshold))[0] + start_idx
            
            if len(peak_indices) > 0:
                peak_temps, peak_derivs = t_vals[peak_indices], derivative[peak_indices]
                tm_val = float(np.sum(peak_temps * peak_derivs) / np.sum(peak_derivs))
            else: tm_val = float(t_vals[max_idx])
        else:
            window = get_dynamic_window(len(f_vals))
            smoothed_f = savgol_filter(f_vals, window_length=window, polyorder=3)
            derivative = np.gradient(smoothed_f, t_vals)
            start_idx, end_idx = int(len(derivative) * 0.1), int(len(derivative) * 0.9)
            local_max_idx = np.argmax(derivative[start_idx:end_idx])
            max_idx = local_max_idx + start_idx
            
            if 0 < max_idx < len(derivative) - 1:
                x1, x2, x3 = t_vals[max_idx-1], t_vals[max_idx], t_vals[max_idx+1]
                y1, y2, y3 = derivative[max_idx-1], derivative[max_idx], derivative[max_idx+1]
                denom = (y1 - 2*y2 + y3)
                if denom != 0:
                    x_peak = x2 - (x2 - x1) * (y3 - y1) / (2 * denom)
                    tm_val = float(x_peak) if x1 <= x_peak <= x3 else float(x2)
                else: tm_val = float(x2)
            else: tm_val = float(t_vals[max_idx])
            
        return t_vals, f_vals, smoothed_f, derivative, tm_val
    except: return None, None, None, None, np.nan

def calculate_tm(temp_series, fluor_series, use_advanced_fitting):
    _, _, _, _, tm = get_curve_arrays(temp_series, fluor_series, use_advanced_fitting)
    return tm

@st.cache_data
def parse_instrument_data_cached(file_content):
    try: df = pd.read_excel(io.BytesIO(file_content), sheet_name="Melt Curve Raw Data", header=None)
    except:
        try: df = pd.read_excel(io.BytesIO(file_content), header=None, engine='xlrd')
        except: df = pd.read_excel(io.BytesIO(file_content), header=None, engine='openpyxl')
    header_row = 0
    for i, row in df.iterrows():
        row_values = [str(val) for val in row.values]
        if any('Well' in val for val in row_values) and any('Temperature' in val for val in row_values):
            header_row = i; break
    df.columns = [str(col).strip() for col in df.iloc[header_row]]
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    
    temp_col, fluor_col = 'Temperature', 'Fluorescence'
    well_col = 'Well Position' if 'Well Position' in df.columns else 'Well'
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[fluor_col] = pd.to_numeric(df[fluor_col], errors='coerce')
    df = df.dropna(subset=[temp_col, fluor_col, well_col])
    df[well_col] = df[well_col].astype(str).apply(lambda x: re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', x.strip().upper().replace(" ", "")))
    wide_df = df.pivot_table(index=temp_col, columns=well_col, values=fluor_col, aggfunc='mean').reset_index()
    return wide_df[temp_col], {col: col for col in wide_df.columns if col != temp_col}, wide_df

def parse_wells_input(wells_str):
    if pd.isna(wells_str) or not str(wells_str).strip(): return []
    return [re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', w.strip().upper().replace(" ", "")) for w in str(wells_str).split(',') if w.strip()]

def update_excludes_callback(exc_key, new_wells):
    if exc_key not in st.session_state: st.session_state[exc_key] = []
    st.session_state[exc_key] = list(set(st.session_state[exc_key] + new_wells))

def assign_wells_callback(cfg_key, wells, name, role, ratio):
    if not wells: return
    df = st.session_state.get(cfg_key, pd.DataFrame(columns=["Wells (孔位)", "Compound_Name (组别命名)", "Role (角色)", "Ratio (浓度比)"]))
    new_row = {"Wells (孔位)": ", ".join(wells), "Compound_Name (组别命名)": name, "Role (角色)": role, "Ratio (浓度比)": str(ratio).strip()}
    st.session_state[cfg_key] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def assign_makeup_wells_callback(cfg_key, wells, cid, role):
    if not wells: return
    df = st.session_state.get(cfg_key, pd.DataFrame(columns=["Well (孔位)", "Compound_ID (化合物ID)", "Role (角色)"]))
    df = df[~df['Well (孔位)'].isin(wells)]
    new_rows = pd.DataFrame([{"Well (孔位)": w, "Compound_ID (化合物ID)": cid, "Role (角色)": role} for w in wells])
    st.session_state[cfg_key] = pd.concat([df, new_rows], ignore_index=True)
    
def set_control_callback(state_key, wells):
    if wells: st.session_state[state_key] = wells

def extract_ratio_value(ratio_str):
    nums = re.findall(r'\d+\.?\d*', str(ratio_str))
    return float(nums[-1]) if nums else 999999

def draw_interactive_plate_mod_a(loaded_wells, key_prefix, excludes=[], nc_wells=[], pc_wells=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state: st.session_state[clear_key] = 0
    rows, cols = list('ABCDEFGH')[::-1], list(range(1, 13))
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c); y_coords.append(r); well_names.append(w)
            if w in excludes: colors.append('#E53935')
            elif w in nc_wells: colors.append('#2196F3') 
            elif w in pc_wells: colors.append('#FF9800') 
            elif w in loaded_wells: colors.append('#4CAF50') 
            else: colors.append('#E0E0E0') 
            
    fig = go.Figure(data=go.Scatter(x=x_coords, y=y_coords, mode='markers+text', text=well_names, customdata=well_names, textposition="middle center", marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')), hoverinfo='text'))
    fig.update_layout(title="🖱️ <b>【交互指南】</b>: 框选孔位并在下方指定角色。<br>图例：🔵 <b>NC</b> | 🟠 <b>PC</b> | 🟢 化合物 | 🔴 <b>已剔除</b> | ⚪ 无数据", xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True), yaxis=dict(showgrid=False, zeroline=False, fixedrange=True), dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40))
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 清除选框", key=f"btn_clr_{key_prefix}"): st.session_state[clear_key] += 1; st.rerun()
    with col_plot:
        event = st.plotly_chart(fig, key=f"plate_{key_prefix}_{st.session_state[clear_key]}", on_select="rerun", selection_mode=['points', 'box', 'lasso'])
    return [pt["customdata"] for pt in event.selection.get("points", [])]

def draw_interactive_plate_val(loaded_wells, key_prefix, cfg_df, excludes=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state: st.session_state[clear_key] = 0
    nc_wells, pc_wells, comp_wells = set(), set(), set()
    if not cfg_df.empty:
        for _, row in cfg_df.iterrows():
            ws = parse_wells_input(row['Wells (孔位)'])
            if row['Role (角色)'] == 'Baseline (NC)': nc_wells.update(ws)
            elif row['Role (角色)'] == 'Positive Control': pc_wells.update(ws)
            elif row['Role (角色)'] == 'Compound': comp_wells.update(ws)
            
    rows, cols = list('ABCDEFGH')[::-1], list(range(1, 13))
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c); y_coords.append(r); well_names.append(w)
            if w in excludes: colors.append('#E53935')
            elif w in nc_wells: colors.append('#2196F3')
            elif w in pc_wells: colors.append('#FF9800')
            elif w in comp_wells: colors.append('#4CAF50')
            elif w in loaded_wells: colors.append('#B0BEC5')
            else: colors.append('#F5F5F5')
            
    fig = go.Figure(data=go.Scatter(x=x_coords, y=y_coords, mode='markers+text', text=well_names, customdata=well_names, textposition="middle center", marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')), hoverinfo='text'))
    fig.update_layout(title="🖱️ <b>【圈选并定义】</b>: 框选孔位后在下方点击分配角色。<br>图例：🔵 <b>NC</b> | 🟠 <b>PC</b> | 🟢 <b>实验组</b> | 🔴 <b>已剔除</b> | ⚪ 待分配", xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True), yaxis=dict(showgrid=False, zeroline=False, fixedrange=True), dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40))
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 清除选框", key=f"btn_clr_{key_prefix}"): st.session_state[clear_key] += 1; st.rerun()
    with col_plot:
        event = st.plotly_chart(fig, key=f"plate_{key_prefix}_{st.session_state[clear_key]}", on_select="rerun", selection_mode=['points', 'box', 'lasso'])
    return [pt["customdata"] for pt in event.selection.get("points", [])]

def draw_interactive_plate_makeup(loaded_wells, key_prefix, cfg_df, excludes=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state: st.session_state[clear_key] = 0
    nc_wells, comp_wells = set(), set()
    if not cfg_df.empty:
        for _, row in cfg_df.iterrows():
            w = row['Well (孔位)']
            if row['Role (角色)'] == 'Baseline (NC)': nc_wells.add(w)
            elif row['Role (角色)'] == 'Compound': comp_wells.add(w)

    rows, cols = list('ABCDEFGH')[::-1], list(range(1, 13))
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c); y_coords.append(r); well_names.append(w)
            if w in excludes: colors.append('#E53935')
            elif w in nc_wells: colors.append('#2196F3')
            elif w in comp_wells: colors.append('#4CAF50')
            elif w in loaded_wells: colors.append('#B0BEC5')
            else: colors.append('#F5F5F5')

    fig = go.Figure(data=go.Scatter(x=x_coords, y=y_coords, mode='markers+text', text=well_names, customdata=well_names, textposition="middle center", marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')), hoverinfo='text'))
    fig.update_layout(title="🖱️ <b>【圈选定义补漏板】</b>: 框选孔位后在下方分配。<br>图例：🔵 <b>NC(基线)</b> | 🟢 <b>已映射化合物</b> | 🔴 <b>已剔除</b> | ⚪ 待分配", xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True), yaxis=dict(showgrid=False, zeroline=False, fixedrange=True), dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40))
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 清除选框", key=f"btn_clr_{key_prefix}"): st.session_state[clear_key] += 1; st.rerun()
    with col_plot:
        event = st.plotly_chart(fig, key=f"plate_{key_prefix}_{st.session_state[clear_key]}", on_select="rerun", selection_mode=['points', 'box', 'lasso'])
    return [pt["customdata"] for pt in event.selection.get("points", [])]

def render_realtime_curve_viewer(selected_wells, temps, well_map, clean_df, use_adv_fit=False):
    if not selected_wells:
        st.info("💡 尚未选定。请在上方 96 孔板上进行拖拽或点选。")
        return

    st.subheader(f"🔍 实时预览选定孔位 ({len(selected_wells)} 个)")
    fig_prev = make_subplots(rows=1, cols=2, subplot_titles=("Raw Fluorescence", "Derivative (dF/dT)"))
    has_data = False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, w in enumerate(selected_wells):
        if w in well_map:
            t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[w]], use_adv_fit)
            if t is not None:
                has_data = True
                c = colors[idx % len(colors)]
                fig_prev.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color=c, width=2), name=w, legendgroup=w), row=1, col=1)
                tm_str = f"{tm:.2f}℃" if not np.isnan(tm) else "N/A"
                fig_prev.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color=c, width=2), name=f"{w} (Tm: {tm_str})", legendgroup=w), row=1, col=2)
                if not np.isnan(tm): fig_prev.add_vline(x=tm, line=dict(color=c, width=1.5, dash='dash'), row=1, col=2)

    if has_data:
        fig_prev.update_layout(height=450, hovermode='x unified', plot_bgcolor='white', margin=dict(t=50, b=40))
        fig_prev.update_xaxes(title_text="Temperature (℃)", showline=True, linewidth=1, linecolor='black', showgrid=False)
        fig_prev.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
        if len(selected_wells) > 12: fig_prev.update_layout(showlegend=False) 
        st.plotly_chart(fig_prev, use_container_width=True)
    else: st.warning("⚠️ 你选定的孔位在文件中没有测定数据。")


def run_dsf_module():
    st.title("🧪 DSF 高通量分析综合平台")
    st.info("""
    **📖 使用说明：**
    针对包含 `Melt Curve Raw Data` 工作表的仪器导出 Excel。系统提供初筛、再验证(排板)、多板整合三大模式。
    支持自动剥离高温/低温噪音假阳性峰，执行双重平滑，以及异常孔位智能质控。
    """)
    
    example_path = "assets/example_dsf.png"
    if os.path.exists(example_path):
        with st.expander("👀 点击查看：处理成功后的标准 DSF 分析图 (模块B展示)"):
            st.image(example_path, caption="Standard DSF Dose-Response Output", use_container_width=True)

    with st.sidebar:
        st.header("🔬 DSF 分析设置")
        analysis_mode = st.radio("请选择当前实验模块：", [
            "🎯 模块A: 初筛 (单板模式)", 
            "🧪 模块B: 再验证 (自定义排板)", 
            "🌌 模块C: 高通量多板初筛 (批量整合与补漏)"
        ])
        st.markdown("---")
        st.header("⚙️ 核心算法引擎配置")
        use_advanced_fitting = st.checkbox("🔬 开启峰顶质心拟合 (修复多峰/锯齿化)", value=False, help="开启后将启用双重 S-G 滤波与质心算法，强制平滑微分曲线并修正因杂质/聚集引发的多峰导致的 Tm 偏移。默认关闭。")
        st.markdown("---")
        st.header("📊 图表显示设置")
        show_baseline_line = st.checkbox("显示阴性对照 (NC) 基准线/曲线", value=True)
        show_pc_line = st.checkbox("显示阳性对照 (PC) 参考线/曲线", value=True)
        st.markdown("---")
        st.header("🛡️ 异常数据质控 (QC) 设置")
        auto_qc = st.checkbox("🤖 开启基于数据的自适应智能质控 (Auto QC)", value=True)
        if not auto_qc:
            qc_max_bg = st.number_input("手动设置: 异常高荧光上限", value=400000, step=50000)
            qc_min_amp = st.number_input("手动设置: 最低有效振幅下限", value=30000, step=5000)
        else:
            qc_max_bg, qc_min_amp = None, None
            st.success("自适应 QC 已开启。系统将自动过滤空孔并精准动态计算 QC 阈值。")

    if analysis_mode == "🎯 模块A: 初筛 (单板模式)":
        st.subheader("🎯 单板初筛自动化分析")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1: lib_type = st.selectbox("化合物库类型", ["FDA", "TCM", "NP"])
        with col2: lib_num = st.number_input("化合物库编号", min_value=1, value=1)
        with col3: hit_threshold = st.number_input("Hit 筛选阈值 (℃)", value=0.5, step=0.1)

        uploaded_file = st.file_uploader("上传原始数据 (.xls/.xlsx)", type=["xlsx", "xls", "csv"], key="file_a")

        if uploaded_file:
            fname = uploaded_file.name
            exc_key = f"manual_exc_a_{fname}"
            nc_key = f"nc_wells_a_{fname}"
            pc_key = f"pc_wells_a_{fname}"
            
            if exc_key not in st.session_state: st.session_state[exc_key] = []
            if nc_key not in st.session_state: st.session_state[nc_key] = ["G12"]
            if pc_key not in st.session_state: st.session_state[pc_key] = ["H12"]
                
            temps, well_map, clean_df = parse_instrument_data_cached(uploaded_file.getvalue())
            
            st.markdown("---")
            st.subheader("👀 实时图谱预览器 & 手动调控")
            curr_nc = st.session_state[nc_key]
            curr_pc = st.session_state[pc_key]
            sel_w = draw_interactive_plate_mod_a(list(well_map.keys()), "mod_a", st.session_state[exc_key], curr_nc, curr_pc)
            render_realtime_curve_viewer(sel_w, temps, well_map, clean_df, use_advanced_fitting)
            
            st.markdown("##### ⚙️ 快速指派孔位角色与过滤")
            st.write(f"**当前选定:** {', '.join(sel_w) if sel_w else '无'} &nbsp;|&nbsp; **阴性对照 (NC):** {', '.join(curr_nc)} &nbsp;|&nbsp; **阳性对照 (PC):** {', '.join(curr_pc)}")
            
            c_btn1, c_btn2, c_btn3, c_exc = st.columns([1, 1, 1, 2])
            with c_btn1: st.button("🔵 设为 NC", on_click=set_control_callback, args=(nc_key, sel_w), use_container_width=True)
            with c_btn2: st.button("🟠 设为 PC", on_click=set_control_callback, args=(pc_key, sel_w), use_container_width=True)
            with c_btn3: st.button("🚫 一键剔除", on_click=update_excludes_callback, args=(exc_key, sel_w), use_container_width=True)
            with c_exc: st.multiselect("当前剔除的异常孔位 (可手动撤销):", list(well_map.keys()), key=exc_key, label_visibility="collapsed")
                    
            st.markdown("---")
            if st.button("🚀 执行全板初筛质控与计算", type="primary"):
                with st.spinner("正在全局解析与质控..."):
                    excludes = st.session_state[exc_key]
                    c_max, c_min = calculate_auto_qc_params(clean_df, well_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                    st.session_state.a_qc_params = (c_max, c_min)
                        
                    nc_tm = pc_tm = np.nan
                    nc_tms = []
                    for w in curr_nc:
                        if w in well_map and w not in excludes and check_curve_quality(clean_df[well_map[w]], c_max, c_min)[0]:
                            nc_tms.append(calculate_tm(temps, clean_df[well_map[w]], use_advanced_fitting))
                    if nc_tms: nc_tm = np.mean(nc_tms)
                    
                    pc_tms = []
                    for w in curr_pc:
                        if w in well_map and w not in excludes and check_curve_quality(clean_df[well_map[w]], c_max, c_min)[0]:
                            pc_tms.append(calculate_tm(temps, clean_df[well_map[w]], use_advanced_fitting))
                    if pc_tms: pc_tm = np.mean(pc_tms)
                        
                    valid_res, anomaly_res = [], []
                    for r in list('ABCDEFGH'):
                        for c in range(1, 13):
                            w = f"{r}{c}"
                            if w in curr_nc or w in curr_pc: continue
                            cid = f"{lib_type.split()[0]}-{lib_num}-{w}"
                            
                            if w in excludes:
                                anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": "人工剔除 (Manual Filtered)", "Delta Tm (℃)": np.nan})
                                continue
                            if w in well_map:
                                f_data = clean_df[well_map[w]]
                                is_v, reason = check_curve_quality(f_data, c_max, c_min)
                                if is_v:
                                    tm_val = calculate_tm(temps, f_data, use_advanced_fitting)
                                    dtm = tm_val - nc_tm if (pd.notnull(tm_val) and pd.notnull(nc_tm)) else np.nan
                                    valid_res.append({"Well": w, "Compound_ID": cid, "Tm (℃)": np.round(tm_val, 2), "Delta Tm (℃)": np.round(dtm, 2)})
                                else: anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": reason, "Delta Tm (℃)": np.nan})
                            else: anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": "No Data", "Delta Tm (℃)": np.nan})
                                
                    a_anom_df = pd.DataFrame(anomaly_res) if anomaly_res else pd.DataFrame(columns=["Well", "Compound_ID", "QC_Status", "Delta Tm (℃)"])
                    st.session_state.update(a_valid_df=pd.DataFrame(valid_res), a_anomaly_df=a_anom_df, a_nc=nc_tm, a_pc=pc_tm, mod_a_done=True)

        if st.session_state.get('mod_a_done'):
            v_df, a_df = st.session_state.a_valid_df, st.session_state.a_anomaly_df
            c_max, c_min = st.session_state.a_qc_params
            st.subheader("📊 初筛质控分析报告")
            st.info(f"💡 当前板执行的 QC 阈值 -> **自发荧光上限**: {c_max:.0f} | **有效振幅下限**: {c_min:.0f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("阴性对照 Tm (基线)", f"{st.session_state.a_nc:.2f} ℃" if pd.notnull(st.session_state.a_nc) else "人工剔除/异常")
            m2.metric("阳性对照 Tm", f"{st.session_state.a_pc:.2f} ℃" if pd.notnull(st.session_state.a_pc) else "人工剔除/异常")
            m3.metric("通过 QC 孔位", len(v_df))
            m4.metric("异常/拦截 (含人工)", len(a_df))
            
            st.subheader("🎯 质控后化合物迁移散点图")
            if not v_df.empty:
                vp = v_df.dropna(subset=["Delta Tm (℃)"])
                xj = np.random.uniform(-1, 1, size=len(vp))
                h = vp[vp["Delta Tm (℃)"] >= hit_threshold]
                nh = vp[vp["Delta Tm (℃)"] < hit_threshold]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xj[vp["Delta Tm (℃)"]<hit_threshold], y=nh["Delta Tm (℃)"], mode='markers', marker=dict(color='#B0BEC5', size=8), text=nh["Compound_ID"], hoverinfo='text+y'))
                fig.add_trace(go.Scatter(x=xj[vp["Delta Tm (℃)"]>=hit_threshold], y=h["Delta Tm (℃)"], mode='markers', marker=dict(color='#E53935', size=12, line_width=1.5, line_color='black'), text=h["Compound_ID"], hoverinfo='text+y'))
                if show_baseline_line: fig.add_hline(y=0, line_color="black", line_width=2, annotation_text="Baseline (NC)", annotation_position="bottom right", annotation_font_size=12)
                if show_pc_line and pd.notnull(st.session_state.a_pc) and pd.notnull(st.session_state.a_nc):
                    pc_dtm = st.session_state.a_pc - st.session_state.a_nc
                    fig.add_hline(y=pc_dtm, line_dash="dot", line_color="#FF9800", line_width=2, annotation_text=f"PC Line ({pc_dtm:.2f}℃)", annotation_position="top right", annotation_font_size=12, annotation_font_color="#FF9800")
                fig.add_hline(y=hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({hit_threshold}℃)")
                fig.update_layout(height=450, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", plot_bgcolor='white', showlegend=False)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks='outside', tickwidth=2, ticklen=6)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(v_df.sort_values(by="Delta Tm (℃)", ascending=False), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 异常数据审查与判读")
            view_anomaly_df = a_df[a_df['QC_Status'] != 'No Data'] if not a_df.empty else pd.DataFrame()
            if not view_anomaly_df.empty:
                view_col1, view_col2 = st.columns([1, 2])
                with view_col1: st.dataframe(view_anomaly_df[['Well', 'Compound_ID', 'QC_Status']], hide_index=True)
                with view_col2:
                    sel_well = st.selectbox("选择要复核的异常孔位:", view_anomaly_df['Well'].tolist(), key="mod_a_anom")
                    if sel_well:
                        temps_a, well_map_a, clean_df_a = parse_instrument_data_cached(uploaded_file.getvalue())
                        reason = view_anomaly_df[view_anomaly_df['Well'] == sel_well].iloc[0]['QC_Status']
                        if sel_well in well_map_a:
                            t, f, s, d, tm = get_curve_arrays(temps_a, clean_df_a[well_map_a[sel_well]], use_advanced_fitting)
                            if t is not None:
                                title_reason = "Manual Filtered" if "人工剔除" in reason else ("Auto-Fluor" if "自发荧光" in reason else "Flat/Quenched")
                                fig_a = make_subplots(rows=1, cols=2, subplot_titles=(f"Raw Fluor [{sel_well}] - {title_reason}", "Derivative - (Excluded)"))
                                fig_a.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
                                fig_a.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color='purple', width=2)), row=1, col=2)
                                fig_a.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                                for annotation in fig_a['layout']['annotations']: annotation['font'] = dict(color='red')
                                fig_a.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                                fig_a.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                                st.plotly_chart(fig_a, use_container_width=True)
            else: st.success("🎉 当前板没有检测到淬灭或高自发荧光的异常数据！")

    elif analysis_mode == "🧪 模块B: 再验证 (自定义排板)":
        st.subheader("🧪 化合物结合验证自动化分析")
        uploaded_files = st.file_uploader("支持批量上传", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="files_b")
        
        if uploaded_files:
            st.markdown("---")
            st.subheader("👀 跨板实时预览与过滤")
            preview_file_name = st.selectbox("选择要配置的孔板文件:", [f.name for f in uploaded_files])
            preview_file = next(f for f in uploaded_files if f.name == preview_file_name)
            
            cfg_key, exc_key = f"val_cfg_{preview_file_name}", f"manual_exc_b_{preview_file_name}"
            if cfg_key not in st.session_state: st.session_state[cfg_key] = pd.DataFrame(columns=["Wells (孔位)", "Compound_Name (组别命名)", "Role (角色)", "Ratio (浓度比)"])
            if exc_key not in st.session_state: st.session_state[exc_key] = []
            
            p_temps, p_well_map, p_clean_df = parse_instrument_data_cached(preview_file.getvalue())
            sel_w = draw_interactive_plate_val(list(p_well_map.keys()), f"mod_b_{preview_file_name}", st.session_state[cfg_key], st.session_state[exc_key])
            render_realtime_curve_viewer(sel_w, p_temps, p_well_map, p_clean_df, use_advanced_fitting)
            
            st.markdown("##### 📝 快速定义选定孔位")
            st.write(f"**当前选定:** {', '.join(sel_w) if sel_w else '无 (请在上方网格中框选)'}")
            
            c1, c2 = st.columns(2)
            c_name = c1.text_input("组别命名 (Compound Name):", key=f"n_b_{preview_file_name}")
            c_ratio = c2.text_input("摩尔浓度比 (Ratio, 留空或填0代表对照):", value="1:1", key=f"r_b_{preview_file_name}")
            
            b1, b2, b3, b4 = st.columns(4)
            with b1: st.button("✅ 设为实验组", on_click=assign_wells_callback, args=(cfg_key, sel_w, c_name, "Compound", c_ratio), use_container_width=True)
            with b2: st.button("🔵 设为 NC", on_click=assign_wells_callback, args=(cfg_key, sel_w, c_name if c_name else "DMSO_NC", "Baseline (NC)", "0"), use_container_width=True)
            with b3: st.button("🟠 设为 PC", on_click=assign_wells_callback, args=(cfg_key, sel_w, c_name if c_name else "Positive_Ctrl", "Positive Control", "0"), use_container_width=True)
            with b4: st.button("🚫 设为异常剔除", on_click=update_excludes_callback, args=(exc_key, sel_w), use_container_width=True)
                
            st.markdown("---")
            col_export, col_import = st.columns(2)
            with col_export:
                st.markdown("###### 📤 导出配置模板")
                if not st.session_state[cfg_key].empty:
                    csv_data = st.session_state[cfg_key].to_csv(index=False).encode('utf-8-sig')
                    st.download_button("一键下载当前配置 (.csv)", data=csv_data, file_name=f"{preview_file_name}_layout.csv", mime="text/csv", use_container_width=True)
                else: st.button("一键下载当前配置 (.csv)", disabled=True, use_container_width=True)
                    
            with col_import:
                st.markdown("###### 📥 导入已有配置")
                uploaded_cfg = st.file_uploader("上传配置 (.csv)", type=["csv"], key=f"up_cfg_{preview_file_name}", label_visibility="collapsed")
                if uploaded_cfg:
                    if st.button("🚨 确认覆盖当前配置", use_container_width=True):
                        try:
                            imp_df = pd.read_csv(uploaded_cfg).fillna("")
                            req_cols = ["Wells (孔位)", "Compound_Name (组别命名)", "Role (角色)", "Ratio (浓度比)"]
                            if all(c in imp_df.columns for c in req_cols):
                                imp_df['Ratio (浓度比)'] = imp_df['Ratio (浓度比)'].astype(str)
                                st.session_state[cfg_key] = imp_df[req_cols]
                                st.rerun()
                            else: st.error("列名不匹配，请确保使用系统导出的模板文件！")
                        except Exception as e: st.error(f"读取配置失败: {str(e)}")

            st.session_state[cfg_key] = st.data_editor(
                st.session_state[cfg_key], 
                column_config={"Wells (孔位)": st.column_config.TextColumn(required=True), "Compound_Name (组别命名)": st.column_config.TextColumn(required=True), "Role (角色)": st.column_config.SelectboxColumn(options=["Baseline (NC)", "Positive Control", "Compound", "Ignore"], required=True), "Ratio (浓度比)": st.column_config.TextColumn()},
                num_rows="dynamic", use_container_width=True, key=f"ed_b_{preview_file_name}"
            )
                    
            if st.button("🚀 计算验证结果", type="primary"):
                with st.spinner("分析中..."):
                    all_res, anom_res, b_qc_records, rep_nc_curve, rep_pc_curve = [], [], [], None, None
                    
                    for file in uploaded_files:
                        cfg = st.session_state.get(f"val_cfg_{file.name}", pd.DataFrame())
                        t, w_map, c_df = parse_instrument_data_cached(file.getvalue())
                        excludes = st.session_state.get(f"manual_exc_b_{file.name}", [])
                        c_max, c_min = calculate_auto_qc_params(c_df, w_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                        b_qc_records.append({'File': file.name, 'Max_BG': c_max, 'Min_Amp': c_min})
                        
                        nc_tms = []
                        if not cfg.empty and 'Role (角色)' in cfg.columns:
                            for w in parse_wells_input(','.join(cfg[cfg['Role (角色)']=='Baseline (NC)']['Wells (孔位)'].astype(str))):
                                if w in excludes: 
                                    anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': '人工剔除'})
                                    continue
                                if w in w_map:
                                    is_v, rsn = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                    if is_v:
                                        t_arr, f_arr, s_arr, d_arr, tm_single = get_curve_arrays(t, c_df[w_map[w]], use_advanced_fitting)
                                        nc_tms.append(tm_single)
                                        if rep_nc_curve is None: rep_nc_curve = (t_arr, d_arr, tm_single)
                                    else: anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': rsn})
                                else: anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': '无数据'})
                                    
                        base = np.mean(nc_tms) if nc_tms else np.nan
                        
                        if not cfg.empty and 'Role (角色)' in cfg.columns:
                            for _, r in cfg[cfg['Role (角色)'].isin(['Compound', 'Positive Control'])].iterrows():
                                cn, role_str, ratio = r['Compound_Name (组别命名)'], r['Role (角色)'], r['Ratio (浓度比)']
                                for w in parse_wells_input(r['Wells (孔位)']):
                                    if w in excludes:
                                        anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': '人工剔除'})
                                        continue
                                    if w in w_map:
                                        is_v, rsn = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                        if is_v: 
                                            t_arr, f_arr, s_arr, d_arr, tm_single = get_curve_arrays(t, c_df[w_map[w]], use_advanced_fitting)
                                            dtm = np.round(tm_single - base, 2) if pd.notnull(base) else np.nan
                                            if role_str == 'Positive Control' and rep_pc_curve is None: rep_pc_curve = (t_arr, d_arr, tm_single)
                                            all_res.append({'File': file.name, 'Compound': cn, 'Role': role_str, 'Ratio': ratio, 'Well': w, 'Tm(℃)': np.round(tm_single, 2), 'Delta_Tm(℃)': dtm, 't_array': t_arr, 'd_array': d_arr, 'raw_tm': tm_single})
                                        else: anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': rsn})
                    
                    b_anom_df = pd.DataFrame(anom_res) if anom_res else pd.DataFrame(columns=['File', 'Compound', 'Well', 'QC_Status'])
                    b_res_df = pd.DataFrame(all_res) if all_res else pd.DataFrame(columns=['File', 'Compound', 'Role', 'Ratio', 'Well', 'Tm(℃)', 'Delta_Tm(℃)', 't_array', 'd_array', 'raw_tm'])
                    st.session_state.update(b_res=b_res_df, b_anom=b_anom_df, b_qc_recs=b_qc_records, b_rep_nc=rep_nc_curve, b_rep_pc=rep_pc_curve, mod_b_done=True)

        if st.session_state.get('mod_b_done'):
            st.subheader("📋 验证分析成果")
            res_df, anom_df = st.session_state.b_res, st.session_state.b_anom
            st.info("💡 系统已执行自适应智能质控。")
            
            if not res_df.empty:
                plot_df = res_df[res_df['Role'] == 'Compound'].copy() if 'Role' in res_df.columns else res_df.copy()
                if not plot_df.empty:
                    plot_df['sort_key'] = plot_df['Ratio'].apply(extract_ratio_value)
                    unique_compounds, npg_colors = plot_df['Compound'].unique(), ['#4DBBD5', '#00A087', '#E64B35', '#F39B7F', '#8491B4', '#91D1C2']
                    
                    use_delta = plot_df['Delta_Tm(℃)'].notna().any()
                    y_col = 'Delta_Tm(℃)' if use_delta else 'Tm(℃)'
                    global_y_min, global_y_max = plot_df[y_col].min(), plot_df[y_col].max()
                    y_margin = (global_y_max - global_y_min) * 0.15 if global_y_max != global_y_min else 0.5
                    global_y_range = [global_y_min - y_margin, global_y_max + y_margin]

                    all_tms_for_x = plot_df['raw_tm'].dropna().tolist()
                    if show_baseline_line and st.session_state.b_rep_nc: all_tms_for_x.append(st.session_state.b_rep_nc[2])
                    if show_pc_line and st.session_state.get('b_rep_pc'): all_tms_for_x.append(st.session_state.b_rep_pc[2])
                    x_axis_range = [np.median(all_tms_for_x) - 12, np.median(all_tms_for_x) + 12] if all_tms_for_x else [30, 90]
                    
                    st.subheader("📈 衍生物熔解曲线阵列 (Derivative Curves Profiles)")
                    cols_cnt = 4
                    rows_cnt = (len(unique_compounds) - 1) // cols_cnt + 1
                    
                    fig_curves = make_subplots(rows=rows_cnt, cols=cols_cnt, subplot_titles=unique_compounds, shared_yaxes=False)
                    fig_trends = make_subplots(rows=rows_cnt, cols=cols_cnt, subplot_titles=unique_compounds, shared_yaxes=False)
                    added_leg_ratios = set()
                    
                    for idx, comp in enumerate(unique_compounds):
                        r, c = idx // cols_cnt + 1, idx % cols_cnt + 1
                        comp_df = plot_df[plot_df['Compound'] == comp].sort_values('sort_key')
                        
                        if show_baseline_line and st.session_state.b_rep_nc:
                            t_nc, d_nc, _ = st.session_state.b_rep_nc
                            show_nc_leg = "Apo (Baseline)" not in added_leg_ratios
                            fig_curves.add_trace(go.Scatter(x=t_nc, y=d_nc, mode='lines', line=dict(color='#333333', dash='dash', width=2), name="Apo (Baseline)", showlegend=show_nc_leg), row=r, col=c)
                            added_leg_ratios.add("Apo (Baseline)")
                            
                        if show_pc_line and st.session_state.get('b_rep_pc'):
                            t_pc, d_pc, _ = st.session_state.b_rep_pc
                            show_pc_leg = "Positive Control" not in added_leg_ratios
                            fig_curves.add_trace(go.Scatter(x=t_pc, y=d_pc, mode='lines', line=dict(color='#FF9800', dash='dashdot', width=2), name="Positive Control", showlegend=show_pc_leg), row=r, col=c)
                            added_leg_ratios.add("Positive Control")
                            
                        trend_data = []
                        for j, ratio in enumerate(comp_df['Ratio'].unique()):
                            ratio_df = comp_df[comp_df['Ratio'] == ratio]
                            rep_row = ratio_df.iloc[0] 
                            c_color, leg_name = npg_colors[j % len(npg_colors)], f"{ratio}"
                            show_leg = leg_name not in added_leg_ratios
                            
                            fig_curves.add_trace(go.Scatter(x=rep_row['t_array'], y=rep_row['d_array'], mode='lines', line=dict(color=c_color, width=2.5), name=leg_name, showlegend=show_leg, legendgroup=leg_name), row=r, col=c)
                            added_leg_ratios.add(leg_name)
                            peak_idx = np.argmax(rep_row['d_array'])
                            fig_curves.add_trace(go.Scatter(x=[rep_row['t_array'][peak_idx]], y=[rep_row['d_array'][peak_idx]], mode='markers', marker=dict(color=c_color, symbol='cross', size=8), showlegend=False, legendgroup=leg_name), row=r, col=c)
                            trend_data.append({'Ratio': str(ratio), 'Mean_Delta': ratio_df[y_col].mean()})
                        
                        if trend_data:
                            t_df = pd.DataFrame(trend_data)
                            comp_color = npg_colors[idx % len(npg_colors)]
                            fig_trends.add_hline(y=0, line_color="black", row=r, col=c)
                            fig_trends.add_trace(go.Scatter(x=t_df['Ratio'], y=t_df['Mean_Delta'], mode='lines+markers+text', text=t_df['Mean_Delta'].apply(lambda x: f"{x:.1f}"), textposition='top center', marker=dict(size=8, color=comp_color, line=dict(width=1, color='black')), line=dict(width=2, color=comp_color), showlegend=False), row=r, col=c)
                    
                    fig_curves.update_layout(height=300 * rows_cnt, plot_bgcolor='white', margin=dict(t=40, r=20, b=20, l=20), legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=13)))
                    fig_trends.update_layout(height=250 * rows_cnt, plot_bgcolor='white', margin=dict(t=40, r=20, b=20, l=20))
                    
                    for i in range(1, rows_cnt * cols_cnt + 1):
                        fig_curves.layout[f'xaxis{i}'].update(showline=True, linecolor='black', showgrid=False, range=x_axis_range)
                        fig_curves.layout[f'yaxis{i}'].update(showline=True, linecolor='black', showgrid=False, showticklabels=False, ticks='')
                        fig_trends.layout[f'xaxis{i}'].update(showline=True, linecolor='black', showgrid=False)
                        fig_trends.layout[f'yaxis{i}'].update(showline=True, linecolor='black', showgrid=True, gridcolor='#EEEEEE', title="ΔTm" if i%cols_cnt==1 else "", range=global_y_range)

                    st.plotly_chart(fig_curves, use_container_width=True)
                    st.subheader("📉 剂量响应趋势阵列 (Dose-Response Trends)")
                    st.plotly_chart(fig_trends, use_container_width=True)
                else: st.warning("⚠️ 当前数据中只包含对照组（或实验组全部被拦截）。")
                    
            st.markdown("##### 📁 验证结果原始数据表")
            if not res_df.empty: st.dataframe(res_df.drop(columns=['Role', 'sort_key', 't_array', 'd_array', 'raw_tm'], errors='ignore'), use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 异常数据审查与判读")
            if not anom_df.empty:
                st.warning(f"⚠️ 发现 {len(anom_df)} 个被质控拦截的异常孔位")
                view_col1, view_col2 = st.columns([1, 2])
                with view_col1: st.dataframe(anom_df, hide_index=True)
                with view_col2:
                    anomaly_opts = [f"[{row['File']}] {row['Well']} ({row['Compound']})" for _, row in anom_df.iterrows()]
                    sel_opt = st.selectbox("选择要复核的异常孔位:", anomaly_opts, key="mod_b_anom_sel")
                    if sel_opt:
                        idx = anomaly_opts.index(sel_opt)
                        target = anom_df.iloc[idx]
                        target_file_obj = next((f for f in uploaded_files if f.name == target['File']), None)
                        if target_file_obj:
                            t_b, w_map_b, c_df_b = parse_instrument_data_cached(target_file_obj.getvalue())
                            if target['Well'] in w_map_b:
                                t, f, s, d, tm = get_curve_arrays(t_b, c_df_b[w_map_b[target['Well']]], use_advanced_fitting)
                                if t is not None:
                                    title_reason = "Manual Filtered" if "人工剔除" in target['QC_Status'] else ("Auto-Fluor" if "自发荧光" in target['QC_Status'] else "Flat/Quenched")
                                    fig_b = make_subplots(rows=1, cols=2, subplot_titles=(f"Raw Fluor [{target['Well']}] - {title_reason}", "Derivative - (Excluded)"))
                                    fig_b.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
                                    fig_b.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color='purple', width=2)), row=1, col=2)
                                    fig_b.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                                    for annotation in fig_b['layout']['annotations']: annotation['font'] = dict(color='red')
                                    fig_b.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                                    fig_b.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                                    st.plotly_chart(fig_b, use_container_width=True)
            else: st.success("🎉 导入的所有复测孔位均完美通过了 QC 质控！")

    elif analysis_mode == "🌌 模块C: 高通量多板初筛 (批量整合与补漏)":
        st.subheader("🌌 高通量多板初筛与补漏整合")
        global_hit_threshold = st.number_input("全局 Hit 筛选阈值 (ΔTm ℃)", value=0.5, step=0.1)
        
        uploaded_files = st.file_uploader("批量上传各板测定数据 (.xls/.xlsx)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="files_c")
        
        if uploaded_files:
            st.markdown("---")
            st.subheader("👀 多板配置中心")
            preview_file_name = st.selectbox("当前配置面板 (切换文件):", [f.name for f in uploaded_files], key="c_sel_file")
            preview_file = next(f for f in uploaded_files if f.name == preview_file_name)
            
            p_type_key = f"c_ptype_{preview_file_name}"
            if p_type_key not in st.session_state: st.session_state[p_type_key] = "🔲 标准规则板"
            p_type = st.radio("该板类型", ["🔲 标准规则板", "🧩 自定义补漏板 (交互式单孔定义)"], key=p_type_key, horizontal=True)

            temps, well_map, clean_df = parse_instrument_data_cached(preview_file.getvalue())

            if "标准" in p_type:
                st.info("此板按 A1-H12 顺序自动生成化合物 ID（将自动忽略指定的 NC 与 H12 阳性孔）。")
                c1, c2, c3 = st.columns(3)
                c1.selectbox("所属库类型", ["FDA", "TCM", "NP", "Custom"], key=f"lt_{preview_file_name}")
                c2.number_input("所属板编号", min_value=1, value=1, key=f"ln_{preview_file_name}")
                c3.text_input("阴性对照 (NC) 孔位", value="G12", key=f"nc_{preview_file_name}")
                
                exc_key = f"manual_exc_c_std_{preview_file_name}"
                if exc_key not in st.session_state: st.session_state[exc_key] = []
                st.multiselect("不参与计算的剔除孔位：", list(well_map.keys()), key=exc_key)
                
            else:
                cfg_key, exc_key = f"c_mkp_cfg_{preview_file_name}", f"manual_exc_c_mkp_{preview_file_name}"
                if cfg_key not in st.session_state: st.session_state[cfg_key] = pd.DataFrame(columns=["Well (孔位)", "Compound_ID (化合物ID)", "Role (角色)"])
                if exc_key not in st.session_state: st.session_state[exc_key] = []

                sel_w = draw_interactive_plate_makeup(list(well_map.keys()), f"mod_c_{preview_file_name}", st.session_state[cfg_key], st.session_state[exc_key])
                render_realtime_curve_viewer(sel_w, temps, well_map, clean_df, use_advanced_fitting)
                
                c_name = st.text_input("化合物 ID / 命名:", placeholder="框选多孔赋予同名时，可随后在下方表格内双击单独修改", key=f"c_name_{preview_file_name}")
                
                cb1, cb2, cb3 = st.columns(3)
                with cb1: st.button("✅ 设为实验组化合物", on_click=assign_makeup_wells_callback, args=(cfg_key, sel_w, c_name, "Compound"), use_container_width=True)
                with cb2: st.button("🔵 设为阴性对照 (NC)", on_click=assign_makeup_wells_callback, args=(cfg_key, sel_w, "NC_Baseline", "Baseline (NC)"), use_container_width=True)
                with cb3: st.button("🚫 设为异常剔除", on_click=update_excludes_callback, args=(exc_key, sel_w), use_container_width=True)

                st.session_state[cfg_key] = st.data_editor(st.session_state[cfg_key], num_rows="dynamic", use_container_width=True, key=f"ed_c_{preview_file_name}")

            st.markdown("---")
            if st.button("🚀 执行多板混合解析与全局统计", type="primary"):
                with st.spinner("正在逐板执行独立 QC 与基线运算..."):
                    master_valid, master_qc_recs, master_pc_dtms = [], [], []
                    for file in uploaded_files:
                        f_ptype = st.session_state.get(f"c_ptype_{file.name}", "🔲 标准规则板")
                        t, w_map, c_df = parse_instrument_data_cached(file.getvalue())
                        excludes = st.session_state.get(f"manual_exc_c_std_{file.name}", []) if "标准" in f_ptype else st.session_state.get(f"manual_exc_c_mkp_{file.name}", [])
                        c_max, c_min = calculate_auto_qc_params(c_df, w_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                        master_qc_recs.append({'File': file.name, 'Plate_Type': "标准板" if "标准" in f_ptype else "补漏板", 'Max_BG': int(c_max), 'Min_Amp': int(c_min)})
                        
                        nc_tm = np.nan
                        if "标准" in f_ptype:
                            nc_w = st.session_state.get(f"nc_{file.name}", "G12")
                            if nc_w in w_map and nc_w not in excludes and check_curve_quality(c_df[w_map[nc_w]], c_max, c_min)[0]:
                                nc_tm = calculate_tm(t, c_df[w_map[nc_w]], use_advanced_fitting)
                            pc_w = "H12"
                            if pc_w in w_map and pc_w not in excludes and check_curve_quality(c_df[w_map[pc_w]], c_max, c_min)[0] and pd.notnull(nc_tm):
                                master_pc_dtms.append(calculate_tm(t, c_df[w_map[pc_w]], use_advanced_fitting) - nc_tm)
                        else:
                            cfg_df = st.session_state.get(f"c_mkp_cfg_{file.name}", pd.DataFrame())
                            nc_tms = []
                            if not cfg_df.empty and 'Role (角色)' in cfg_df.columns:
                                nc_wells = cfg_df[cfg_df['Role (角色)'] == 'Baseline (NC)']['Well (孔位)'].tolist()
                                for nc_w in nc_wells:
                                    if nc_w in w_map and nc_w not in excludes and check_curve_quality(c_df[w_map[nc_w]], c_max, c_min)[0]:
                                        nc_tms.append(calculate_tm(t, c_df[w_map[nc_w]], use_advanced_fitting))
                                if nc_tms: nc_tm = np.mean(nc_tms)
                        
                        if "标准" in f_ptype:
                            l_type, l_num, nc_w = st.session_state.get(f"lt_{file.name}", "FDA"), st.session_state.get(f"ln_{file.name}", 1), st.session_state.get(f"nc_{file.name}", "G12")
                            for r in list('ABCDEFGH'):
                                for c in range(1, 13):
                                    w = f"{r}{c}"
                                    if w in [nc_w, "H12"] or w in excludes: continue
                                    cid = f"{l_type.split()[0]}-{l_num}-{w}"
                                    if w in w_map:
                                        is_v, _ = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                        if is_v:
                                            tm_val = calculate_tm(t, c_df[w_map[w]], use_advanced_fitting)
                                            master_valid.append({'File': file.name, 'Well': w, 'Compound_ID': cid, 'Tm (℃)': tm_val, 'Delta_Tm (℃)': (tm_val - nc_tm if pd.notnull(nc_tm) else np.nan)})
                        else:
                            cfg_df = st.session_state.get(f"c_mkp_cfg_{file.name}", pd.DataFrame())
                            if not cfg_df.empty and 'Role (角色)' in cfg_df.columns:
                                for _, row in cfg_df[cfg_df['Role (角色)'] == 'Compound'].iterrows():
                                    w, cid = row['Well (孔位)'], row['Compound_ID (化合物ID)']
                                    if w not in excludes and w in w_map:
                                        is_v, _ = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                        if is_v:
                                            tm_val = calculate_tm(t, c_df[w_map[w]], use_advanced_fitting)
                                            master_valid.append({'File': file.name, 'Well': w, 'Compound_ID': cid, 'Tm (℃)': tm_val, 'Delta_Tm (℃)': (tm_val - nc_tm if pd.notnull(nc_tm) else np.nan)})
                    
                    st.session_state.update(c_res=pd.DataFrame(master_valid), c_qc=pd.DataFrame(master_qc_recs), c_pc_dtm=(np.mean(master_pc_dtms) if master_pc_dtms else np.nan), mod_c_done=True)

        if st.session_state.get('mod_c_done'):
            c_res = st.session_state.c_res
            st.subheader("🌐 全局筛选成果 (HTS Global Overview)")
            
            valid_hits = c_res[c_res['Delta_Tm (℃)'] >= global_hit_threshold] if not c_res.empty else pd.DataFrame()
            c1, c2, c3 = st.columns(3)
            c1.metric("解析总有效化合物", len(c_res))
            c2.metric(f"检出 Hit 总数 (≥{global_hit_threshold}℃)", len(valid_hits))
            c3.metric("整体阳性率", f"{len(valid_hits)/len(c_res)*100:.2f} %" if not c_res.empty else "0 %")
            
            st.subheader("🌌 全局化合物基线偏移散点图")
            if not c_res.empty:
                vp = c_res.dropna(subset=['Delta_Tm (℃)']).copy()
                vp['Hover_Text'] = vp['Compound_ID'] + "<br>File: " + vp['File'] + " | Well: " + vp['Well']
                xj = np.random.uniform(-1, 1, size=len(vp))
                h = vp[vp["Delta_Tm (℃)"] >= global_hit_threshold]
                nh = vp[vp["Delta_Tm (℃)"] < global_hit_threshold]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xj[vp["Delta_Tm (℃)"]<global_hit_threshold], y=nh["Delta_Tm (℃)"], mode='markers', marker=dict(color='#B0BEC5', size=6, opacity=0.6), name='Background', text=nh['Hover_Text'], hoverinfo='text+y'))
                fig.add_trace(go.Scatter(x=xj[vp["Delta_Tm (℃)"]>=global_hit_threshold], y=h["Delta_Tm (℃)"], mode='markers', marker=dict(color='#E64B35', size=10, line_width=1.5, line_color='black'), name='Hits', text=h['Hover_Text'], hoverinfo='text+y'))
                
                if show_baseline_line: fig.add_hline(y=0, line_color="black", line_width=2, annotation_text="Baseline (NC)", annotation_position="bottom right", annotation_font_size=12)
                c_pc = st.session_state.get('c_pc_dtm', np.nan)
                if show_pc_line and pd.notnull(c_pc): fig.add_hline(y=c_pc, line_dash="dot", line_color="#FF9800", line_width=2, annotation_text=f"Mean PC Line ({c_pc:.2f}℃)", annotation_position="top right", annotation_font_size=12, annotation_font_color="#FF9800")
                fig.add_hline(y=global_hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Global Thr ({global_hit_threshold}℃)")
                
                fig.update_layout(height=500, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", plot_bgcolor='white', hovermode='closest', font=dict(family="Arial", size=14, color="black"))
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks='outside', tickwidth=2, ticklen=6, showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
                
            st.dataframe(c_res.sort_values(by="Delta_Tm (℃)", ascending=False), use_container_width=True)


# =====================================================================
#                          主程序侧边栏路由
# =====================================================================
def main():
    st.sidebar.title("🧬 数据处理工具箱")
    st.sidebar.markdown("请在下方选择所需功能模块：")
    
    app_mode = st.sidebar.radio("模块导航", [
        "📊 模块 1：晶体衍射数据汇总表", 
        "🔬 模块 2：CBF 衍射图像查看器",
        "📈 模块 3：DSF 熔解曲线分析"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 结构生物学自动化中心")

    if app_mode == "📊 模块 1：晶体衍射数据汇总表":
        run_table1_module()
    elif app_mode == "🔬 模块 2：CBF 衍射图像查看器":
        run_cbf_module()
    elif app_mode == "📈 模块 3：DSF 熔解曲线分析":
        run_dsf_module()

if __name__ == "__main__":
    main()
