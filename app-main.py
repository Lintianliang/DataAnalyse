import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
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
    - **2. Phenix 精修日志 (必传)**：精修完成后的 `.log` 文本。用于提取 Rwork/Rfree、原子数量、B 因子等模型质量指标。
    - **3. 最终 PDB 结构文件 (推荐)**：即提交至数据库前的 `.pdb` 文件。包含最权威的晶胞参数及拉氏图数据。（*若未上传，将智能尝试从前两份日志中降级抠取备用数据*）。
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
                    st.toast("💡 未检测到独立 PDB，已自动从 Phenix log 尾部提取结构参数。")

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
