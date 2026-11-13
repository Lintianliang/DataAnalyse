# coding=utf-8
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
import subprocess
import os
import xml.etree.ElementTree as ET
import gemmi
import tempfile
import glob
import math
import base64

# ================= UI 设置 =================
st.set_page_config(page_title="ComplexViewer Pro", layout="wide")
st.title("🧬 ComplexViewer Pro - 蛋白与配体互作分析平台 (期刊发表级)")

# ================= 侧边栏：数据输入 =================
st.sidebar.header("📁 1. 数据输入")
pdb_file = st.sidebar.file_uploader("上传复合物 PDB 文件", type=["pdb"])
density_file = st.sidebar.file_uploader("上传原始衍射电子密度图 (.mtz/.ccp4/.map)", type=["mtz", "ccp4", "map"])
st.sidebar.info("💡 提示：现已整合 PyMOL 云端处理引擎。直接上传全局 .mtz 文件，系统将在后台自动精准剥离配体周围的局域密度网格，彻底告别全局噪点遮挡！")

st.sidebar.markdown("---")
st.sidebar.header("🔧 2. 配体残基识别")
target_ligand = st.sidebar.text_input("小分子残基名 (如 LIG, JZ4):", value="LIG").upper().strip()

# ================= 侧边栏：分析设置 (严谨标准) =================
st.sidebar.markdown("---")
st.sidebar.header("🔗 3. 互作力分析控制")
st.sidebar.info("📌 本平台已统一采用学术界公认的 **PLIP** 严谨算法，确保互作力判定符合顶刊发表要求。")

show_hp = st.sidebar.checkbox("⬜ 显示疏水作用 (Hydrophobic)", value=True)
show_pi = st.sidebar.checkbox("🟧 显示 π-π 堆积 (Pi-Stacking)", value=True)
show_sb = st.sidebar.checkbox("🟥 显示盐桥 (Salt-Bridges)", value=True)
show_wb = st.sidebar.checkbox("💧 显示水桥 (Water-Bridges)", value=True)
show_halo = st.sidebar.checkbox("🟪 显示卤键 (Halogen-Bonds)", value=True)

st.sidebar.markdown("---")
show_res_labels = st.sidebar.checkbox("🏷️ 显示互作氨基酸标签", value=True)
show_distance = st.sidebar.checkbox("📏 显示连线距离测定", value=True)

# ================= 侧边栏：视图风格切换 =================
st.sidebar.markdown("---")
st.sidebar.header("🎨 4. 期刊级视图风格")
style_mode = st.sidebar.radio("选择显示风格 (Style):", [
    "🌟 经典科研风 (Classic)", 
    "📝 JACS / JMC 极简风 (Minimalist)", 
    "🧪 Nature / Cell 高级黑 (Dark Mode)",
    "🧊 纯净白底水墨风 (Clean White)"
])

bg_color, prot_color, prot_style, prot_opacity = "white", "gray", "cartoon", 0.7
lig_color, pocket_color = "cyanCarbon", "whiteCarbon"
label_bg, label_color, mesh_color = "white", "black", "blue"

if style_mode == "📝 JACS / JMC 极简风 (Minimalist)":
    prot_color, prot_style, prot_opacity = "lightgray", "line", 1.0
    lig_color, pocket_color, mesh_color = "orangeCarbon", "greenCarbon", "#3366ff"
elif style_mode == "🧪 Nature / Cell 高级黑 (Dark Mode)":
    bg_color, prot_color, prot_opacity = "#121212", "#444444", 0.9
    lig_color, pocket_color, mesh_color = "greenCarbon", "grayCarbon", "cyan"
    label_bg, label_color = "black", "white"
elif style_mode == "🧊 纯净白底水墨风 (Clean White)":
    prot_color, prot_opacity = "white", 1.0
    lig_color, pocket_color, mesh_color = "yellowCarbon", "tealCarbon", "darkblue"

# ================= 侧边栏：视角控制 =================
st.sidebar.markdown("---")
st.sidebar.header("🎯 5. 快速视角定位")
view_focus = st.sidebar.radio("选择聚焦:", ["🧪 当前位点配体", "🕳️ 当前结合口袋", "🌍 全局复合物"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.header("☁️ 6. 电子密度网格 (PyMOL 引擎)")
show_density = st.sidebar.checkbox("显示已裁切的局域密度网格", value=True)
carve_radius = st.sidebar.slider("PyMOL 裁切半径 (Carve, Å)", min_value=1.0, max_value=4.0, value=1.6, step=0.1)
isolevel = st.sidebar.slider("网格阈值 (Sigma / Level)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

# ================= 核心分析引擎 =================
def fix_pdb_ligand(pdb_string, lig_name):
    if not lig_name: return pdb_string
    lines = pdb_string.split('\n')
    fixed = []
    for line in lines:
        if line.startswith('ATOM  ') or line.startswith('HETATM'):
            if line[17:20].strip() == lig_name:
                line = 'HETATM' + line[6:]
        fixed.append(line)
    return '\n'.join(fixed)

def get_ligand_chains(pdb_string, lig_name):
    if not lig_name: return []
    chains = set()
    for line in pdb_string.split('\n'):
        if (line.startswith('ATOM  ') or line.startswith('HETATM')) and len(line) >= 22:
            if line[17:20].strip() == lig_name: chains.add(line[21].strip())
    return sorted(list(chains))

@st.cache_data
def run_plip(pdb_string):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "complex.pdb")
        with open(pdb_path, "w", encoding="utf-8") as f: f.write(pdb_string)
        
        wrapper_path = os.path.join(tmpdir, "run_plip_custom.py")
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(f"import sys\nfrom plip.plipcmd import main\nsys.argv = ['plip', '-f', r'{pdb_path}', '-x', '-o', r'{tmpdir}']\nmain()\n")
        
        res = subprocess.run(["python", wrapper_path], capture_output=True, text=True, encoding="utf-8", errors="replace")
        xml_files = glob.glob(os.path.join(tmpdir, "*.xml"))
        if xml_files: return ET.parse(xml_files[0]).getroot()
        return None

def parse_interactions(xml_root, target_chain):
    interactions = []
    if xml_root is None: return interactions
    def get_lbl(node):
        try: return f"{node.find('restype').text}{node.find('resnr').text}"
        except: return ""
    for site in xml_root.findall('.//bindingsite'):
        identifiers = site.find('identifiers')
        if identifiers is not None and target_chain:
            s_chain = identifiers.find('chain')
            if s_chain is not None and s_chain.text and s_chain.text.strip() != target_chain: continue
        for hb in site.findall('.//hydrogen_bonds/hb'):
            lig_coo = [float(hb.find(f'ligcoo/{a}').text) for a in ('x', 'y', 'z')]
            prot_coo = [float(hb.find(f'protcoo/{a}').text) for a in ('x', 'y', 'z')]
            interactions.append(('H-Bond', lig_coo, prot_coo, 'blue', get_lbl(hb)))
        for hy in site.findall('.//hydrophobic_interactions/hydrophobic_interaction'):
            lig_coo = [float(hy.find(f'ligcoo/{a}').text) for a in ('x', 'y', 'z')]
            prot_coo = [float(hy.find(f'protcoo/{a}').text) for a in ('x', 'y', 'z')]
            interactions.append(('Hydrophobic', lig_coo, prot_coo, 'gray', get_lbl(hy)))
        for pi in site.findall('.//pi_stacks/pi_stack'):
            lig_coo = [float(pi.find(f'ligcoo/{a}').text) for a in ('x', 'y', 'z')]
            prot_coo = [float(pi.find(f'protcoo/{a}').text) for a in ('x', 'y', 'z')]
            interactions.append(('Pi-Stack', lig_coo, prot_coo, 'orange', get_lbl(pi)))
        for sb in site.findall('.//salt_bridges/salt_bridge'):
            lig_coo = [float(sb.find(f'ligcoo/{a}').text) for a in ('x', 'y', 'z')]
            prot_coo = [float(sb.find(f'protcoo/{a}').text) for a in ('x', 'y', 'z')]
            interactions.append(('Salt-Bridge', lig_coo, prot_coo, 'red', get_lbl(sb)))
        for wb in site.findall('.//water_bridges/water_bridge'):
            lig_coo = [float(wb.find(f'ligcoo/{a}').text) for a in ('x', 'y', 'z')]
            prot_coo = [float(wb.find(f'protcoo/{a}').text) for a in ('x', 'y', 'z')]
            interactions.append(('Water-Bridge', lig_coo, prot_coo, 'cyan', get_lbl(wb)))
        for hlx in site.findall('.//halogen_bonds/halogen_bond'):
            lig_coo = [float(hlx.find(f'ligcoo/{a}').text) for a in ('x', 'y', 'z')]
            prot_coo = [float(hlx.find(f'protcoo/{a}').text) for a in ('x', 'y', 'z')]
            interactions.append(('Halogen-Bond', lig_coo, prot_coo, 'purple', get_lbl(hlx)))
    return interactions

# ================= 核心创新：调用后台 PyMOL 引擎进行物理裁切 =================
@st.cache_data
def process_density_with_pymol(pdb_bytes, map_bytes, filename, lig_name, radius):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "complex.pdb")
        with open(pdb_path, "wb") as f: f.write(pdb_bytes)
        
        map_path = os.path.join(tmpdir, "density.ccp4")
        if filename.lower().endswith('.mtz'):
            mtz_path = os.path.join(tmpdir, "density.mtz")
            with open(mtz_path, "wb") as f: f.write(map_bytes)
            mtz = gemmi.read_mtz_file(mtz_path)
            try: grid = mtz.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=3)
            except:
                try: grid = mtz.transform_f_phi_to_map('2FOFCWT', 'PH2FOFCWT', sample_rate=3)
                except: return None
            grid.normalize()
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = grid
            ccp4.update_ccp4_header()
            ccp4.write_ccp4_map(map_path)
        else:
            with open(map_path, "wb") as f: f.write(map_bytes)
            
        out_ccp4 = os.path.join(tmpdir, "carved.ccp4")
        
        # 构建并运行 PyMOL 脚本
        pml_script = f"""
load {pdb_path}, complex_obj
load {map_path}, density_map
select my_ligand, resn {lig_name} and complex_obj
map_trim density_map, my_ligand, {radius}
save {out_ccp4}, density_map
quit
"""
        pml_path = os.path.join(tmpdir, "run.pml")
        with open(pml_path, "w", encoding="utf-8") as f: f.write(pml_script)
        
        # 调用无头模式的 PyMOL
        subprocess.run(["pymol", "-c", "-q", pml_path], capture_output=True)
        
        if os.path.exists(out_ccp4):
            with open(out_ccp4, "rb") as f: return f.read()
        return None

# ================= 动态解析与 UI 衔接 =================
raw_pdb_str, clean_pdb_str, target_chain = "", "", ""

if pdb_file is not None:
    raw_pdb_str = pdb_file.getvalue().decode("utf-8", errors="replace")
    clean_pdb_str = fix_pdb_ligand(raw_pdb_str, target_ligand)
    chains = get_ligand_chains(clean_pdb_str, target_ligand)
    
    if chains:
        if len(chains) > 1:
            st.sidebar.markdown("---")
            st.sidebar.header("🎯 配体位点锁定")
            options = [f"Chain {c if c else '_'}" for c in chains]
            choice = st.sidebar.selectbox("⚠️ 发现配体分布在多条链，请选择独立分析的链:", options)
            target_chain = chains[options.index(choice)]
        else:
            target_chain = chains[0]
            st.sidebar.success(f"✅ 已自动锁定唯一链: Chain {target_chain if target_chain else '_'}")

if pdb_file is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"🖥️ 3D 结构交互视窗 - {style_mode.split()[0]}")
        
        view = py3Dmol.view(width=900, height=700)
        view.setBackgroundColor(bg_color)
        view.addModel(clean_pdb_str, 'pdb')
        
        if prot_style == "cartoon":
            if style_mode == "🧊 纯净白底水墨风 (Clean White)": view.setStyle({'protein': True}, {'cartoon': {'color': 'white', 'opacity': 1.0, 'style': 'outline'}})
            else: view.setStyle({'protein': True}, {'cartoon': {'color': prot_color, 'opacity': prot_opacity}})
        else:
            view.setStyle({'protein': True}, {'line': {'color': prot_color}})
        
        ligand_sel = {'resn': target_ligand}
        if target_chain: ligand_sel['chain'] = target_chain
        if not target_ligand: ligand_sel = {'not': {'or': [{'protein': True}, {'resn': 'HOH'}, {'resn': 'WAT'}]}}
        pocket_sel = {'within': {'distance': 5.0, 'sel': ligand_sel}}
        
        if style_mode == "🧊 纯净白底水墨风 (Clean White)": view.addStyle(pocket_sel, {'stick': {'colorscheme': pocket_color, 'radius': 0.12}})
        else: view.addStyle(pocket_sel, {'stick': {'colorscheme': pocket_color, 'radius': 0.15}})
        view.setStyle(ligand_sel, {'stick': {'colorscheme': lig_color, 'radius': 0.25}})
        
        xml_root = run_plip(clean_pdb_str)
        raw_interactions = parse_interactions(xml_root, target_chain)
        
        final_interactions = []
        for attr in raw_interactions:
            itype, lig_coo, prot_coo, color, lbl = attr
            if itype == 'H-Bond': final_interactions.append(('H-Bond', lig_coo, prot_coo, 'blue', lbl))
            elif itype == 'Hydrophobic' and show_hp: final_interactions.append(attr)
            elif itype == 'Pi-Stack' and show_pi: final_interactions.append(attr)
            elif itype == 'Salt-Bridge' and show_sb: final_interactions.append(attr)
            elif itype == 'Water-Bridge' and show_wb: final_interactions.append(attr)
            elif itype == 'Halogen-Bond' and show_halo: final_interactions.append(attr)
        
        for attr in final_interactions:
            itype, lig_coo, prot_coo, color, lbl = attr
            view.addCylinder({'start': {'x': lig_coo[0], 'y': lig_coo[1], 'z': lig_coo[2]}, 'end': {'x': prot_coo[0], 'y': prot_coo[1], 'z': prot_coo[2]}, 'radius': 0.08 if style_mode != "📝 JACS / JMC 极简风 (Minimalist)" else 0.04, 'color': color, 'dashed': True})
            if show_res_labels and lbl:
                view.addLabel(lbl, {'position': {'x': prot_coo[0], 'y': prot_coo[1], 'z': prot_coo[2]}, 'fontColor': label_color, 'backgroundColor': label_bg, 'backgroundOpacity': 0.8, 'fontSize': 12 if style_mode != "📝 JACS / JMC 极简风 (Minimalist)" else 10, 'inFront': True, 'showBackground': True, 'borderThickness': 1 if style_mode == "🧊 纯净白底水墨风 (Clean White)" else 0, 'borderColor': "black"})
            if show_distance:
                dist = math.sqrt(sum((l - p)**2 for l, p in zip(lig_coo, prot_coo)))
                view.addLabel(f"{dist:.1f} Å", {'position': {'x': (lig_coo[0]+prot_coo[0])/2, 'y': (lig_coo[1]+prot_coo[1])/2, 'z': (lig_coo[2]+prot_coo[2])/2}, 'fontColor': color, 'backgroundColor': label_bg, 'backgroundOpacity': 0.8, 'fontSize': 11})
        
        if view_focus == "🧪 当前位点配体": view.zoomTo(ligand_sel)
        elif view_focus == "🕳️ 当前结合口袋": view.zoomTo(pocket_sel)
        else: view.zoomTo()

        # ================= 调用 PyMOL 生成纯净局域网格 =================
        map_data_b64 = ""
        if density_file is not None and show_density:
            with st.spinner(f"🚀 正在后台调用 PyMOL 引擎切取配体 {carve_radius} Å 内纯净电子云..."):
                map_data = process_density_with_pymol(
                    pdb_file.getvalue(), density_file.getvalue(), density_file.name, target_ligand, carve_radius
                )
                if map_data: map_data_b64 = base64.b64encode(map_data).decode("utf-8")
                else: st.warning("⚠️ 密度图解析失败，请确保云端环境已配置 packages.txt")

        # ================= HTML / JS 截屏与网格渲染注入 =================
        html_str = view._make_html()
        
        density_js = ""
        if map_data_b64:
            density_js = f"""
            <script>
            function loadDensity() {{
                var viewer = null;
                for (var key in window) {{
                    if (key.indexOf('viewer_') === 0 && window[key] && typeof window[key].pngURI === 'function') {{ viewer = window[key]; break; }}
                }}
                if (!viewer) {{ setTimeout(loadDensity, 200); return; }}
                try {{
                    var binary_string = window.atob("{map_data_b64}");
                    var len = binary_string.length;
                    var bytes = new Uint8Array(len);
                    for (var i = 0; i < len; i++) {{ bytes[i] = binary_string.charCodeAt(i); }}
                    // 核心：直接渲染 PyMOL 切割好的极纯净 map，应用镂空文献风格！
                    var volspec = {{isoval: {isolevel}, color: "{mesh_color}", opacity: 0.6, wireframe: true}};
                    viewer.addVolumetricData(bytes.buffer, "ccp4", volspec);
                    viewer.render();
                }} catch (err) {{ console.error("密度渲染错误: " + err.message); }}
            }}
            loadDensity();
            </script>
            """
            
        export_html = f"""
        <div style="position: absolute; top: 15px; left: 15px; z-index: 2147483647; pointer-events: auto;">
            <button onclick="downloadPNGWithLegend()" style="padding: 10px 15px; font-size: 14px; font-weight: bold; background-color: #ff4b4b; color: white; border: 2px solid white; border-radius: 6px; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.4);">📸 导出带图例高清图</button>
        </div>
        <script>
        function downloadPNGWithLegend() {{
            try {{
                var viewer = null;
                for (var key in window) {{
                    if (key.indexOf('viewer_') === 0 && window[key] && typeof window[key].pngURI === 'function') {{ viewer = window[key]; break; }}
                }}
                if (!viewer) {{ alert('3D 渲染器未就绪，请稍候...'); return; }}
                var imgURI = viewer.pngURI();
                var img = new Image();
                img.onload = function() {{
                    var canvas = document.createElement('canvas');
                    canvas.width = img.width; canvas.height = img.height;
                    var ctx = canvas.getContext('2d');
                    ctx.fillStyle = "{bg_color}";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                    var boxW = 230, boxH = 170, legendX = 20, legendY = canvas.height - boxH - 20;
                    ctx.fillStyle = "{'rgba(255, 255, 255, 0.9)' if label_bg == 'white' else 'rgba(30, 30, 30, 0.9)'}";
                    ctx.fillRect(legendX, legendY, boxW, boxH);
                    ctx.strokeStyle = "{'#444' if label_color == 'black' else '#ccc'}";
                    ctx.lineWidth = 1; ctx.strokeRect(legendX, legendY, boxW, boxH);
                    ctx.font = "bold 15px sans-serif"; ctx.fillStyle = "{label_color}";
                    ctx.fillText("Interactions Legend", legendX + 15, legendY + 25);
                    var legends = [{{color: "blue", text: "H-Bond"}}, {{color: "gray", text: "Hydrophobic"}}, {{color: "orange", text: "Pi-Stacking"}}, {{color: "red", text: "Salt-Bridge"}}, {{color: "cyan", text: "Water-Bridge"}}, {{color: "purple", text: "Halogen-Bond"}}];
                    ctx.font = "14px sans-serif";
                    for(var i=0; i<legends.length; i++) {{
                        var y = legendY + 55 + i * 20;
                        ctx.beginPath(); ctx.setLineDash([5, 5]); ctx.moveTo(legendX + 15, y - 4); ctx.lineTo(legendX + 45, y - 4);
                        ctx.strokeStyle = legends[i].color; ctx.lineWidth = 3; ctx.stroke();
                        ctx.fillStyle = "{label_color}"; ctx.fillText(legends[i].text, legendX + 60, y);
                    }}
                    var finalURI = canvas.toDataURL("image/png", 1.0);
                    var a = document.createElement('a'); a.download = 'ComplexViewer_Export.png'; a.href = finalURI;
                    document.body.appendChild(a); a.click(); setTimeout(function() {{ document.body.removeChild(a); }}, 100);
                }};
                img.src = imgURI;
            }} catch (err) {{ alert("截图失败: " + err.message); }}
        }}
        </script>
        """
        
        inject_html = export_html + density_js
        if "<body>" in html_str: html_str = html_str.replace("<body>", "<body>\n" + inject_html)
        else: html_str = inject_html + html_str
        components.html(html_str, width=900, height=700)

    with col2:
        st.subheader("📊 单一链位点互作统计")
        counts = {}
        for itype, _, _, _, _ in final_interactions: counts[itype] = counts.get(itype, 0) + 1
        if counts:
            for k, v in counts.items(): st.metric(label=f"{k} 数量", value=v)
        else: st.info("💡 当前链未检测到显著作用力。")
        if xml_root is None: st.error("PLIP 运算失败。")
        st.markdown("---")
        st.markdown('''**🔍 结构图例**\n* 🔘 **飘带**: 蛋白骨架\n* 🤍 **细棒**: 5Å 内口袋残基\n* 🧪 **粗棒**: 小分子配体''')
else:
    st.info("👈 请在左侧上传复合物 PDB 数据以启动分析。")