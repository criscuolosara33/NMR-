import streamlit as st
from streamlit_ketcher import st_ketcher
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import warnings

st.set_page_config(page_title="NMR Laboratory", layout="wide")

# --- CSS E COSTANTI ESTETICHE ---
BORDEAUX = '#6B1422'
BORDEAUX_HOVER = '#822433'

st.markdown(f"""
<style>
    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, table, th, td {{
        font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif !important;
    }}
    div[data-testid="metric-container"] {{
        background-color: #fafafa;
        border: 1px solid #e6e6e6;
        padding: 15px 20px;
        border-radius: 6px;
        box-shadow: 1px 2px 4px rgba(0,0,0,0.04);
    }}
    div.stButton > button:first-child {{ 
        background-color: {BORDEAUX}; 
        color: white; 
        border: none;
        border-radius: 4px;
        font-weight: bold;
        letter-spacing: 0.5px;
        transition: all 0.2s ease-in-out;
    }}
    div.stButton > button:hover {{ 
        background-color: {BORDEAUX_HOVER}; 
        color: white; 
        box-shadow: 0 4px 6px rgba(107, 20, 34, 0.2);
    }}
    hr {{ margin-top: 1.5em; margin-bottom: 1.5em; border-color: #e6e6e6; }}
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Times New Roman', 'serif']

if 'ultimo_smiles' not in st.session_state: st.session_state.ultimo_smiles = ""
if 'stato_app' not in st.session_state: st.session_state.stato_app = "input" 
if 'parametri' not in st.session_state: st.session_state.parametri = {}

# --- FUNZIONI CHIMICHE DI BASE ---
def calcola_proprieta(mol):
    mol_h = Chem.AddHs(mol)
    n_tetra = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [6, 14]) 
    n_tri = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [7, 15]) 
    n_mono = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [1, 9, 17, 35, 53]) 
    dbe = n_tetra + 1 - (n_mono / 2.0) + (n_tri / 2.0)
    return {
        'formula': rdMolDescriptors.CalcMolFormula(mol_h), 
        'mw': Descriptors.MolWt(mol), 
        'dbe': dbe, 
        'formula_dbe_str': rf"n_{{IV}} + 1 - \frac{{n_{{I}}}}{{2}} + \frac{{n_{{III}}}}{{2}}", 
        'formula_dbe_val_str': rf"{n_tetra} + 1 - \frac{{{n_mono}}}{{2}} + \frac{{{n_tri}}}{{2}}", 
        'mol_h': mol_h, 'mol_no_h': mol
    }

def ottieni_nomi_pubchem(smiles):
    try:
        url_iupac = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/property/IUPACName/JSON"
        res_iupac = requests.get(url_iupac, timeout=5)
        iupac = res_iupac.json()['PropertyTable']['Properties'][0]['IUPACName'] if res_iupac.status_code == 200 else "N/D"
        url_syn = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/synonyms/JSON"
        res_syn = requests.get(url_syn, timeout=5)
        comune = res_syn.json()['InformationList']['Information'][0]['Synonym'][0] if res_syn.status_code == 200 else "N/D"
        return iupac, comune
    except Exception: return "Errore connessione", "Errore connessione"

def analizza_stereochimica(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    commenti = []
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if chiral_centers:
        dett_chirali = [f"C{idx + 1} ({c})" if c in ['R', 'S'] else f"C{idx + 1} (Stereochimica Ambicua/Non Definita)" for idx, c in chiral_centers]
        commenti.append(f"**Centri stereogenici**: Rilevati agli atomi {', '.join(dett_chirali)}.")
        ch2_dias = [str(atom.GetIdx() + 1) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2]
        if ch2_dias: commenti.append(f"**Protoni diastereotopici**: I gruppi metilenici ({', '.join(ch2_dias)}) risiedono in un intorno chirale. I protoni sono anisocroni e presentano un accoppiamento geminale attivo ($^2J$).")
    else:
        commenti.append("**Topologia achirale**: La molecola non presenta centri stereogenici definiti. I gruppi metilenici contengono protoni enantiotopici (isocroni in mezzi achirali).")
    
    dett_ez = [f"C{b.GetBeginAtomIdx()+1}=C{b.GetEndAtomIdx()+1} ({'E' if b.GetStereo()==Chem.BondStereo.STEREOE else 'Z'})" 
               for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]]
    if dett_ez: commenti.append(f"**Isomeria geometrica**: Rilevati doppi legami con configurazione {', '.join(dett_ez)}.")
    return commenti

def analizza_simmetria_equivalenza(mol):
    commenti = []
    mol_h = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    gruppi_c, gruppi_h = {}, {}
    for atom in mol_h.GetAtoms():
        r = ranks[atom.GetIdx()]
        if atom.GetAtomicNum() == 6:
            if r not in gruppi_c: gruppi_c[r] = []
            gruppi_c[r].append(str(atom.GetIdx() + 1))
        elif atom.GetAtomicNum() == 1:
            idx_str = str(atom.GetNeighbors()[0].GetIdx() + 1)
            if r not in gruppi_h: gruppi_h[r] = []
            gruppi_h[r].append(idx_str)

    equiv_c = [g for g in gruppi_c.values() if len(g) > 1]
    if equiv_c: commenti.append(f"**Equivalenza Chimica (13C)**: Operazioni di simmetria del primo ordine correlano i carboni: " + " | ".join([f"({', '.join(g)})" for g in equiv_c]) + ".")
    equiv_h = [g for g in gruppi_h.values() if len(g) > 1 and len(set(g)) > 1]
    if equiv_h:
        equiv_h_formattati = [f"({', '.join(set(g))})" for g in equiv_h]
        commenti.append(f"**Equivalenza Chimica (1H)**: Operazioni di simmetria correlano i protoni legati ai carboni: {' | '.join(equiv_h_formattati)}.")
    return commenti

# --- ARCHITETTURA OOP PER SPIN SYSTEM ED EQUIVALENZA ---
class Nucleus:
    def __init__(self, atom_idx, element, shift_base, chem_eq_class, is_exch, attached_c):
        self.id = atom_idx
        self.element = element
        self.shift = shift_base
        self.chem_eq = chem_eq_class
        self.mag_eq = None
        self.is_exchangeable = is_exch
        self.attached_c = attached_c
        self.couplings = {}

class Coupling:
    def __init__(self, id_a, id_b, j_val, order, path_len):
        self.id_a = id_a
        self.id_b = id_b
        self.j_val = j_val
        self.order = order
        self.path_len = path_len

class SpinSystemEngine:
    def __init__(self, mol_h, freq_mhz):
        self.mol = mol_h
        self.freq = freq_mhz
        self.nuclei = {}
        self.couplings = []
        self.spin_systems_graphs = []
        self._build_engine()

    def _stima_shift_base(self, atom):
        neighbor = atom.GetNeighbors()[0]
        if neighbor.GetIsAromatic(): return 7.3
        elif neighbor.GetAtomicNum() == 8: return 4.5
        elif neighbor.GetAtomicNum() == 7: return 2.5
        elif neighbor.GetAtomicNum() == 16: return 1.5
        elif neighbor.GetAtomicNum() == 6:
            if neighbor.GetHybridization() == Chem.HybridizationType.SP2:
                is_aldehyde = any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(neighbor).GetAtomicNum() == 8 for b in neighbor.GetBonds())
                return 9.8 if is_aldehyde else 5.5
            elif neighbor.GetHybridization() == Chem.HybridizationType.SP: return 2.8
            else: return 0.9 + (0.3 * sum(1 for a in neighbor.GetNeighbors() if a.GetAtomicNum() == 6))
        return 2.0

    def _build_engine(self):
        ranks = list(Chem.CanonicalRankAtoms(self.mol, breakTies=False))
        shifts_visti = []
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                idx = atom.GetIdx()
                c_idx = atom.GetNeighbors()[0].GetIdx() + 1 if atom.GetNeighbors()[0].GetAtomicNum() == 6 else None
                is_exch = atom.GetNeighbors()[0].GetAtomicNum() in [7, 8, 16]
                shift = self._stima_shift_base(atom)
                while any(abs(shift - sv) < 0.05 for sv in shifts_visti): shift += 0.1
                shifts_visti.append(shift)
                self.nuclei[idx] = Nucleus(idx, '1H', shift, ranks[idx], is_exch, c_idx)

        h_ids = list(self.nuclei.keys())
        for i in range(len(h_ids)):
            for j in range(i + 1, len(h_ids)):
                n1, n2 = h_ids[i], h_ids[j]
                if self.nuclei[n1].is_exchangeable or self.nuclei[n2].is_exchangeable: continue
                path = Chem.GetShortestPath(self.mol, n1, n2)
                plen = len(path) - 1
                j_val = 0.0
                if plen == 2: j_val = 12.0
                elif plen == 3: j_val = 7.5
                elif plen == 4 and any(self.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in path): j_val = 2.0
                if j_val > 0:
                    self.couplings.append(Coupling(n1, n2, j_val, "first", plen))
                    self.nuclei[n1].couplings[n2] = j_val
                    self.nuclei[n2].couplings[n1] = j_val

        chem_groups = {}
        for nuc in self.nuclei.values():
            chem_groups.setdefault(nuc.chem_eq, []).append(nuc)

        mag_eq_counter = 0
        for eq_class, nucs in chem_groups.items():
            if len(nucs) == 1:
                nucs[0].mag_eq = mag_eq_counter
                mag_eq_counter += 1
                continue
            mag_groups = {}
            for nuc in nucs:
                sig = []
                for target_id, j_val in nuc.couplings.items():
                    if self.nuclei[target_id].chem_eq != eq_class:
                        sig.append((self.nuclei[target_id].chem_eq, j_val))
                nuc_sig = tuple(sorted(sig))
                mag_groups.setdefault(nuc_sig, []).append(nuc)
            for sig, m_nucs in mag_groups.items():
                for mn in m_nucs: mn.mag_eq = mag_eq_counter
                mag_eq_counter += 1

        visited = set()
        for nuc_id in self.nuclei:
            if nuc_id not in visited and not self.nuclei[nuc_id].is_exchangeable:
                system = []
                queue = [nuc_id]
                while queue:
                    curr = queue.pop(0)
                    if curr not in visited:
                        visited.add(curr)
                        system.append(curr)
                        for neighbor in self.nuclei[curr].couplings:
                            if neighbor not in visited: queue.append(neighbor)
                if system: self.spin_systems_graphs.append(system)

    def get_signals_for_ui(self):
        signals = []
        gruppi_mag = {}
        for nuc in self.nuclei.values():
            gruppi_mag.setdefault(nuc.mag_eq, []).append(nuc)
            
        for mag_class, nucs in gruppi_mag.items():
            rep = nucs[0]
            integral = len(nucs)
            if rep.is_exchangeable:
                signals.append(self._format_signal(rep, integral, nucs, 'br s', [], "**Singoletto allargato**: Protone soggetto a scambio chimico rapido; gli accoppiamenti vicinali sono soppressi."))
                continue

            j_vicini = []
            for target_id, j_val in rep.couplings.items():
                if self.nuclei[target_id].mag_eq != mag_class: j_vicini.append(j_val)
            j_vicini.sort(reverse=True)

            commento_ordine = ""
            if rep.chem_eq != rep.mag_eq:
                commento_ordine = "<br><br>- <b>Sistema Second-Order</b>: I nuclei appartengono alla stessa classe di equivalenza chimica ma differiscono per equivalenza magnetica (es. sistemi AA'BB'). Il multipletto reale sarà più complesso."
            else:
                for target_id, j_val in rep.couplings.items():
                    delta_nu = abs(rep.shift - self.nuclei[target_id].shift) * self.freq
                    if delta_nu > 0 and (delta_nu / j_val) < 10:
                        commento_ordine = f"<br><br>- <b>Accoppiamento Forte</b>: Rilevato basso rapporto Δν/J ≈ {delta_nu/j_val:.1f}. Il sistema devia dall'approssimazione del prim'ordine (distorsione di intensità/effetto tetto)."
                        break

            j_details = []
            for j in j_vicini:
                if j == 12.0: j_details.append(f"{j} Hz (Geminale, $^2J$)")
                elif j == 7.5: j_details.append(f"{j} Hz (Vicinale/Orto, $^3J$)")
                elif j == 2.0: j_details.append(f"{j} Hz (Meta/Long-range, $^4J$)")
                else: j_details.append(f"{j} Hz")
            
            j_str = f"<br><br><b>Costanti J rilevate:</b> {', '.join(j_details)}" if j_details else ""

            if not j_vicini: mult = 's'
            elif len(j_vicini) == 1: mult = 'd'
            else:
                counts = {}
                for jv in j_vicini: counts[jv] = counts.get(jv, 0) + 1
                chars = [{1:'d', 2:'t', 3:'q'}.get(num, 'm') for num in counts.values()]
                mult = 'm' if 'm' in chars or sum(counts.values()) > 6 else "".join(chars)

            base_comment = self._descrivi_mult(mult)
            signals.append(self._format_signal(rep, integral, nucs, mult, j_vicini, base_comment + commento_ordine + j_str))
            
        return signals

    def _descrivi_mult(self, mult):
        diz = {'s': "**Singoletto**: Nessun accoppiamento vicinale rilevato.", 
               'd': "**Doppietto**: Accoppiamento con un nucleo magneticamente equivalente.", 
               't': "**Tripletto**: Accoppiamento con due nuclei equivalenti.", 
               'q': "**Quartetto**: Accoppiamento con tre nuclei equivalenti.", 
               'm': "**Multipletto**: Sovrapposizione complessa di stati di spin."}
        if mult in diz: return diz[mult]
        
        nomi = {'d': "Doppietto", 't': "Tripletto", 'q': "Quartetto"}
        plur = {'d': "doppietti", 't': "tripletti", 'q': "quartetti"}
        
        if len(mult) == 2 and all(c in nomi for c in mult): 
            return f"**{nomi[mult[0]]} di {plur[mult[1]]}**: Risoluzione dello splitting tree con costanti J distinte."
        elif len(mult) == 3 and all(c in nomi for c in mult): 
            return f"**{nomi[mult[0]]} di {plur[mult[1]]} di {plur[mult[2]]}**: Splitting tree triplo."
        return "**Multipletto complesso**: Generato dalla cascata di accoppiamenti di prim'ordine multipli."

    def _format_signal(self, rep, integral, nucs, mult, j_vals, comment):
        sig = {
            'delta': rep.shift,
            'multiplicity': mult,
            'integral': integral,
            'atoms': list({n.attached_c for n in nucs if n.attached_c is not None}),
            'h_atoms': [n.id for n in nucs],
            'is_exchangeable': rep.is_exchangeable,
            'coupling_comment': comment,
            'mag_eq': rep.mag_eq,
            'chem_eq': rep.chem_eq
        }
        sig['sub_peaks'] = self._genera_sotto_picchi(sig['delta'], mult, float(integral), self.freq, j_vals)
        return sig

    def _genera_sotto_picchi(self, center, mult, integral, freq, j_vals):
        if mult in ['s', 'br s']: return [(center, integral)]
        if mult == 'm':
            j_std = 7.5 / freq
            return [(center + o, r * integral) for o, r in zip(np.linspace(-1.5*j_std, 1.5*j_std, 5), [0.1, 0.25, 0.3, 0.25, 0.1])]

        def ottieni_offset(carattere, j_val_hz):
            j_ppm = j_val_hz / freq
            if carattere == 'd': return [-j_ppm/2, j_ppm/2], [0.5, 0.5]
            elif carattere == 't': return [-j_ppm, 0, j_ppm], [0.25, 0.5, 0.25]
            elif carattere == 'q': return [-1.5*j_ppm, -0.5*j_ppm, 0.5*j_ppm, 1.5*j_ppm], [0.125, 0.375, 0.375, 0.125]
            return [0.0], [1.0]

        picchi = [(center, integral)]
        chars = [c for c in mult if c in 'dtq']
        for i, c in enumerate(chars):
            j = j_vals[i] if i < len(j_vals) else 7.5
            nuovi_picchi = []
            off, rat = ottieni_offset(c, j)
            for p_shift, p_int in picchi:
                for o, r in zip(off, rat): nuovi_picchi.append((p_shift + o, p_int * r))
            picchi = nuovi_picchi
        return picchi

def stima_locale_13c(mol_no_h):
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, breakTies=False))
    groups = {}
    for atom in mol_no_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            r = ranks[atom.GetIdx()]
            groups.setdefault(r, []).append(atom)

    signals, shifts_visti = [], []
    for r, c_atoms in groups.items():
        rep_c = c_atoms[0]
        n_h_attached = rep_c.GetTotalNumHs()
        
        shift = 30.0
        n_neighbors_C = sum(1 for n in rep_c.GetNeighbors() if n.GetAtomicNum() == 6)
        n_neighbors_O = sum(1 for n in rep_c.GetNeighbors() if n.GetAtomicNum() == 8)
        n_neighbors_N = sum(1 for n in rep_c.GetNeighbors() if n.GetAtomicNum() == 7)

        if rep_c.GetHybridization() == Chem.HybridizationType.SP2:
            if rep_c.GetIsAromatic(): shift = 130.0
            elif any(mol_no_h.GetBondBetweenAtoms(rep_c.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE and n.GetAtomicNum() == 8 for n in rep_c.GetNeighbors()): shift = 170.0
            else: shift = 120.0
        elif rep_c.GetHybridization() == Chem.HybridizationType.SP: shift = 70.0
        else: shift += (n_neighbors_C * 8) + (n_neighbors_O * 40) + (n_neighbors_N * 20)

        while any(abs(shift - sv) < 0.5 for sv in shifts_visti): shift += 0.5
        shifts_visti.append(shift)
        
        tipo_c = "Cq" if n_h_attached == 0 else f"CH{n_h_attached}" if n_h_attached > 1 else "CH"
        signals.append({'delta': shift, 'multiplicity': 's', 'integral': len(c_atoms), 'atoms': [atom.GetIdx() + 1 for atom in c_atoms], 'n_h': n_h_attached, 'tipo_c': tipo_c, 'is_exchangeable': False, 'coupling_comment': f"**Singoletto disaccoppiato**: Modello Broadband ($^{{13}}$C{{$^{{1H}}$}}). Natura del nucleo: {tipo_c}"})
    return signals

def salva_pagina_uniforme(pdf, fig):
    fig.set_size_inches(11.69, 8.27) 
    pdf.savefig(fig, orientation='landscape', bbox_inches='tight')
    plt.close(fig)

# --- UI MAIN ---
st.title("NMR Laboratory (Interactive Platform)")

smiles = st_ketcher()

if smiles != st.session_state.ultimo_smiles:
    st.session_state.ultimo_smiles = smiles
    st.session_state.stato_app = "input"

if st.session_state.stato_app == "input":
    st.markdown("### Impostazioni Strumento (Acquisizione)")
    c1, c2 = st.columns(2)
    freq_1h = c1.selectbox("Frequenza di Lavoro (MHz)", [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0], index=2)
    solv_1h = c2.selectbox("Solvente (Deuterato)", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
    
    st.markdown("### Modalità 13C-NMR")
    c3, c4, c5 = st.columns(3)
    freq_13c = c3.selectbox("Frequenza 13C (MHz)", [75.0, 100.0, 125.0, 150.0, 200.0, 250.0], index=2)
    solv_13c = c4.selectbox("Solvente 13C", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
    modo_13c = c5.selectbox("Esperimento a Impulsi", ["Broadband", "DEPT-135", "DEPT-90", "APT"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    cb1, cb2, cb3 = st.columns(3)
    
    if cb1.button("Acquisisci Spettro 1H", use_container_width=True):
        if not smiles: st.error("Disegna una molecola prima di procedere.")
        else:
            st.session_state.parametri = {'freq': freq_1h, 'solvente': solv_1h, 'tech': '1h'}
            st.session_state.stato_app = "calcolo_1h"
            st.rerun()
            
    if cb2.button("Acquisisci Spettro 13C", use_container_width=True):
        if not smiles: st.error("Disegna una molecola prima di procedere.")
        else:
            st.session_state.parametri = {'freq': freq_13c, 'solvente': solv_13c, 'tech': modo_13c}
            st.session_state.stato_app = "calcolo_13c"
            st.rerun()
            
    if cb3.button("Mappa COSY 2D", use_container_width=True):
        if not smiles: st.error("Disegna una molecola prima di procedere.")
        else:
            st.session_state.parametri = {'freq': freq_1h, 'solvente': solv_1h, 'tech': 'cosy'}
            st.session_state.stato_app = "calcolo_cosy"
            st.rerun()

elif st.session_state.stato_app in ["calcolo_1h", "calcolo_13c", "calcolo_cosy"]:
    
    if st.button("← Ritorna ai Parametri Strumentali", use_container_width=False):
        st.session_state.stato_app = "input"
        st.rerun()
        
    mol = Chem.MolFromSmiles(st.session_state.ultimo_smiles)
    if mol is None: st.error("Struttura non valida. Verifica gli atomi e i legami in Ketcher.")
    else:
        props = calcola_proprieta(mol)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Formula Molecolare", props['formula'])
        c2.metric("Massa Molare", f"{props['mw']:.2f} g/mol")
        c3.metric("DBE (Insaturazioni)", f"{props['dbe']:.1f}")
        
        st.caption("La formula per il DBE considera gli atomi tetravalenti (C, Si) + 1, sottrae la metà dei monovalenti (H, alogeni) e aggiunge la metà dei trivalenti (N, P). Gli atomi bivalenti (O, S) non influenzano il computo formale.")

        p = st.session_state.parametri
        if st.session_state.stato_app == 'calcolo_1h':
            freq, solv, tech, nmr_type, plot_title, x_range = p['freq'], p['solvente'], '1h', '1h', f'Spettro 1H-NMR ({int(p["freq"])} MHz, {p["solvente"]})', [-0.5, 12.5]
            engine = SpinSystemEngine(props['mol_h'], freq)
            signals = engine.get_signals_for_ui()
        elif st.session_state.stato_app == 'calcolo_13c':
            freq, solv, tech, nmr_type, plot_title, x_range = p['freq'], p['solvente'], p['tech'], '13c', f'Spettro 13C-NMR [{p["tech"]}] ({int(p["freq"])} MHz, {p["solvente"]})', [-10, 220]
            signals = stima_locale_13c(props['mol_no_h'])
        else:
            freq, solv, tech, nmr_type, plot_title, x_range = p['freq'], p['solvente'], 'cosy', 'cosy', f'Correlazione Omonucleare COSY 2D ({int(p["freq"])} MHz, {p["solvente"]})', [-0.5, 12.5]
            engine = SpinSystemEngine(props['mol_h'], freq)
            signals = engine.get_signals_for_ui()

        with st.expander("🔬 Analisi Strutturale e Sistema di Spin"):
            for commento in analizza_simmetria_equivalenza(mol): st.markdown(commento)
            for commento in analizza_stereochimica(mol): st.markdown(commento)

            if nmr_type in ['1h', 'cosy']:
                num_chem = len(set([n.chem_eq for n in engine.nuclei.values() if not n.is_exchangeable]))
                num_mag = len(set([n.mag_eq for n in engine.nuclei.values() if not n.is_exchangeable]))
                st.write(f"- **Classi di Equivalenza**: {num_chem} gruppi isocroni chimicamente; {num_mag} gruppi magneticamente equivalenti.")
                st.write(f"- **Sistemi di Spin**: Individuate {len(engine.spin_systems_graphs)} reti di spin mutuamente accoppiate e indipendenti.")

        # Pre-calcolo delle coordinate spettro 1D
        x_ppm = np.linspace(x_range[0], x_range[1], int(freq * 200))
        gamma_base = 0.0025 * (500.0 / freq) if nmr_type == '1h' else 0.5
        y_intensity = np.zeros_like(x_ppm)
        segnali_visibili = []

        for sig in signals:
            if nmr_type == '1h':
                scambiato = (solv in ["D2O", "CD3OD"] and sig.get('is_exchangeable', False))
                if scambiato: continue 
                segnali_visibili.append(sig)
                gamma_app = 0.06 if sig.get('is_exchangeable', False) else gamma_base
                for p_shift, p_int in sig['sub_peaks']: y_intensity += p_int / (1.0 + ((x_ppm - p_shift) / gamma_app)**2)
            elif nmr_type == '13c':
                n_h = sig.get('n_h', 0)
                if tech == "DEPT-135": p_int = -1.0 if n_h == 2 else (0.0 if n_h == 0 else 1.0)
                elif tech == "DEPT-90": p_int = 1.0 if n_h == 1 else 0.0
                elif tech == "APT": p_int = 1.0 if n_h in [0, 2] else -1.0
                else: p_int = 1.0
                
                if p_int != 0.0:
                    segnali_visibili.append(sig)
                    y_intensity += p_int / (1.0 + ((x_ppm - float(sig.get('delta', 1.0))) / gamma_base)**2)

        y_min = min(y_intensity) * 1.15 if min(y_intensity) < 0 else 0
        y_max = max(y_intensity) * 1.15 if np.any(y_intensity) else 1

        # --- PREPARAZIONE DATI PER TABELLA PULITA ---
        df_data = []
        original_comments = {} 
        
        for sig in signals:
            scambiato = (nmr_type == '1h' and solv in ["D2O", "CD3OD"] and sig.get('is_exchangeable', False))
            scomparso_dept = False
            note_acc = sig['coupling_comment']
            
            if nmr_type == '13c':
                n_h = sig.get('n_h', 0)
                if tech == "DEPT-135" and n_h == 0: scomparso_dept = True; note_acc = "Nucleo quaternario, il segnale collassa nel DEPT-135."
                if tech == "DEPT-90" and n_h != 1: scomparso_dept = True; note_acc = "Nucleo non terziario, segnale soppresso nel DEPT-90."
            if scambiato: note_acc = f"Protone scambiato attivamente in solvente deuterato {solv}."
            
            shift_val = float(sig['delta'])
            original_comments[shift_val] = note_acc
            
            row = {
                'Shift (ppm)': "N/D" if scambiato or scomparso_dept else f"{shift_val:.2f}",
                'Molteplicità': sig['multiplicity'] if not (scambiato or scomparso_dept) else "-",
                'Atomi': ", ".join(map(str, sig['atoms'])),
                '_sort_val': shift_val
            }
            if nmr_type == '1h': row['Integrale'] = sig['integral'] if not scambiato else "-"
            else: row['Tipo'] = sig.get('tipo_c', 'C')
            
            df_data.append(row)
            
        df_signals_display = pd.DataFrame(df_data).sort_values(by='_sort_val', ascending=False)
        cols_order = ['Shift (ppm)', 'Integrale' if nmr_type == '1h' else 'Tipo', 'Molteplicità', 'Atomi']
        df_signals_clean = df_signals_display[cols_order]

        # --- LOGICA PLOT COSY 2D ---
        if nmr_type == 'cosy':
            st.markdown("---")
            cross_peaks_idx = set()
            for i, sigA in enumerate(signals):
                for j, sigB in enumerate(signals):
                    if i >= j: continue
                    coupled = False
                    for hA in sigA['h_atoms']:
                        for hB in sigB['h_atoms']:
                            path = Chem.GetShortestPath(props['mol_h'], hA, hB)
                            if len(path) == 4: coupled = True; break
                        if coupled: break
                    if coupled: cross_peaks_idx.add((i, j))
            
            n_pts = 600
            x_grid = np.linspace(x_range[0], x_range[1], n_pts)
            X, Y = np.meshgrid(x_grid, x_grid)
            Z = np.zeros_like(X)
            
            gamma_2d = 0.015 * (500.0 / freq)
            gamma_1d = 0.0025 * (500.0 / freq)

            for sig in signals:
                if not sig.get('is_exchangeable', False):
                    for p_shift, p_int in sig['sub_peaks']:
                        Z += p_int / (1.0 + ((X - p_shift)/gamma_2d)**2 + ((Y - p_shift)/gamma_2d)**2)
                        
            for i, j in cross_peaks_idx:
                sigA, sigB = signals[i], signals[j]
                for px, px_int in sigA['sub_peaks']:
                    for py, py_int in sigB['sub_peaks']:
                        Z += (px_int * py_int * 0.3) / (1.0 + ((X - px)/gamma_2d)**2 + ((Y - py)/gamma_2d)**2)
                        Z += (px_int * py_int * 0.3) / (1.0 + ((X - py)/gamma_2d)**2 + ((Y - px)/gamma_2d)**2)

            fig_cosy = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8], vertical_spacing=0.01)
            fig_cosy.add_trace(go.Scatter(x=x_ppm, y=y_intensity, mode='lines', line=dict(color=BORDEAUX, width=1.5), hoverinfo='skip', showlegend=False), row=1, col=1)
            fig_cosy.add_trace(go.Contour(
                z=Z, x=x_grid, y=x_grid, colorscale=[[0, 'white'], [1, BORDEAUX]], showscale=False,
                contours=dict(start=0.1, size=(np.max(Z) - 0.1) / 8 if np.max(Z) > 0.1 else 1, coloring='lines'),
                line=dict(width=1.0), hoverinfo='none'
            ), row=2, col=1)
            fig_cosy.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', line=dict(color='rgba(0,0,0,0.2)', dash='dash'), hoverinfo='skip', showlegend=False), row=2, col=1)
            
            fig_cosy.update_layout(title=plot_title, width=800, height=900, plot_bgcolor='white', font=dict(family="Palatino, serif"), margin=dict(l=40, r=40, t=60, b=40))
            fig_cosy.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            fig_cosy.update_xaxes(autorange="reversed", showgrid=True, gridcolor='#E0E0E0', row=1, col=1)
            fig_cosy.update_xaxes(title_text="Chemical Shift δ (ppm)", autorange="reversed", showgrid=True, gridcolor='#E0E0E0', row=2, col=1)
            fig_cosy.update_yaxes(title_text="Chemical Shift δ (ppm)", autorange="reversed", scaleanchor="x2", scaleratio=1, showgrid=True, gridcolor='#E0E0E0', row=2, col=1)
            
            st.plotly_chart(fig_cosy, use_container_width=True)
            
        # --- LOGICA PLOT 1D E TABELLA INTERATTIVA ---
        else:
            st.markdown("---")
            col_table, col_mol = st.columns([0.6, 0.4])
            
            with col_table:
                st.markdown("### Assegnazione Segnali (Clicca una riga)")
                event = st.dataframe(df_signals_clean, use_container_width=True, selection_mode="single-row", on_select="rerun")
            
            selected_atoms, selected_delta, selected_mult = [], None, ""
            long_comment = ""
            width_box = 0
            
            if len(event.selection.rows) > 0:
                idx = event.selection.rows[0]
                row_data = df_signals_clean.iloc[idx]
                atomi_str = row_data['Atomi']
                if atomi_str != "N/D" and atomi_str != "": 
                    selected_atoms = [int(a) - 1 for a in atomi_str.split(", ")]
                try: 
                    selected_delta = float(row_data['Shift (ppm)'])
                    selected_mult = row_data['Molteplicità']
                    long_comment = original_comments.get(selected_delta, "")
                except ValueError: selected_delta = None

            with col_mol:
                st.markdown("### Nuclei Responsabili")
                fig_highlight = plt.figure(dpi=300, figsize=(5, 5))
                ax_high = fig_highlight.add_subplot(111)
                for atom in mol.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx() + 1))
                
                selected_bonds = []
                if len(selected_atoms) > 1:
                    for bond in mol.GetBonds():
                        if bond.GetBeginAtomIdx() in selected_atoms and bond.GetEndAtomIdx() in selected_atoms:
                            selected_bonds.append(bond.GetIdx())
                
                d2d_high = rdMolDraw2D.MolDraw2DCairo(1500, 1500)
                opts = d2d_high.drawOptions()
                opts.annotationFontScale = 0.9
                bordeaux_rgba = (107/255, 20/255, 34/255, 0.4)
                highlight_dict = {a: bordeaux_rgba for a in selected_atoms}
                highlight_bonds_dict = {b: bordeaux_rgba for b in selected_bonds}
                opts.setHighlightColour(bordeaux_rgba)
                
                d2d_high.DrawMolecule(mol, highlightAtoms=selected_atoms, highlightAtomColors=highlight_dict, highlightBonds=selected_bonds, highlightBondColors=highlight_bonds_dict)
                d2d_high.FinishDrawing()
                ax_high.imshow(Image.open(io.BytesIO(d2d_high.GetDrawingText())))
                ax_high.axis('off')
                st.pyplot(fig_highlight)
                plt.close(fig_highlight)

            # Pannello Dinamico: Spiegazione dello splitting e Zoom
            if selected_delta is not None:
                st.markdown("---")
                st.markdown(f"### Dettaglio del Segnale a {selected_delta:.2f} ppm")
                c_testo, c_zoom = st.columns([0.65, 0.35])
                with c_testo:
                    st.markdown(f"""
                    <div style="background-color: #f5f5f5; border: 1px solid #d3d3d3; border-left: 5px solid #6c757d; padding: 15px; border-radius: 4px; color: #333333; font-size: 15px; height: 100%;">
                        {long_comment}
                    </div>
                    """, unsafe_allow_html=True)
                with c_zoom:
                    fig_singolo_zoom = plt.figure(figsize=(3, 1.5), dpi=100)
                    ax_zoom = fig_singolo_zoom.add_subplot(111)
                    ax_zoom.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=2.0)
                    molt_f = len(selected_mult) if len(selected_mult) > 0 else 1
                    width_box = (0.05 * molt_f) * (500.0 / freq)
                    width_zoom = width_box * 1.5 
                    ax_zoom.set_xlim(selected_delta + width_zoom, selected_delta - width_zoom)
                    mask = (x_ppm >= selected_delta - width_zoom) & (x_ppm <= selected_delta + width_zoom)
                    ax_zoom.set_ylim(0, (np.max(y_intensity[mask]) if np.any(mask) else 1) * 1.1)
                    ax_zoom.get_yaxis().set_visible(False)
                    for spine in ['top', 'right', 'left']: ax_zoom.spines[spine].set_visible(False)
                    st.pyplot(fig_singolo_zoom)
                    plt.close(fig_singolo_zoom)

            st.markdown("### Simulazione Spettroscopica Globale")
            fig_interattivo = go.Figure()
            fig_interattivo.add_trace(go.Scatter(x=x_ppm, y=y_intensity, mode='lines', line=dict(color=BORDEAUX, width=1.5)))
            if nmr_type == '13c' and tech in ["DEPT-135", "APT"]: fig_interattivo.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
            
            if selected_delta is not None and nmr_type == '1h':
                fig_interattivo.add_vrect(
                    x0=selected_delta + width_box, x1=selected_delta - width_box,
                    fillcolor=BORDEAUX, opacity=0.18, layer="above", line_width=1.5, line_color=BORDEAUX,
                    annotation_text=f"{selected_delta:.2f} ppm", annotation_position="top left"
                )
            
            fig_interattivo.update_layout(title=plot_title, xaxis_title="Chemical Shift δ (ppm)", yaxis_title="Intensità Relativa", xaxis=dict(autorange="reversed"), plot_bgcolor='white', hovermode='x', height=600, font=dict(family="Palatino, serif"))
            fig_interattivo.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
            fig_interattivo.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0', showticklabels=False)
            st.plotly_chart(fig_interattivo, use_container_width=True)

            # --- GENERAZIONE PDF DINAMICO A FINE SCRIPT ---
            pdf_buffer = io.BytesIO()
            with PdfPages(pdf_buffer) as pdf:
                
                # Pagina 1: Struttura Base/Evidenziata
                fig_mol_pdf = plt.figure(dpi=300)
                ax_mol_pdf = fig_mol_pdf.add_subplot(111)
                for atom in mol.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx() + 1))
                d2d_pdf = rdMolDraw2D.MolDraw2DCairo(1500, 1000)
                opts_pdf = d2d_pdf.drawOptions()
                opts_pdf.annotationFontScale = 0.9
                if selected_delta is not None:
                    opts_pdf.setHighlightColour(bordeaux_rgba)
                    d2d_pdf.DrawMolecule(mol, highlightAtoms=selected_atoms, highlightAtomColors=highlight_dict, highlightBonds=selected_bonds, highlightBondColors=highlight_bonds_dict)
                else:
                    d2d_pdf.DrawMolecule(mol)
                d2d_pdf.FinishDrawing()
                ax_mol_pdf.imshow(Image.open(io.BytesIO(d2d_pdf.GetDrawingText())))
                ax_mol_pdf.axis('off')
                salva_pagina_uniforme(pdf, fig_mol_pdf)
                
                # Pagina 2: Tabella Assegnazioni
                fig_tab_pdf = plt.figure(dpi=300)
                ax_tab_pdf = fig_tab_pdf.add_subplot(111)
                ax_tab_pdf.axis('off')
                tab_data = df_signals_clean.astype(str).values.tolist()
                tab_cols = df_signals_clean.columns.tolist()
                tab = ax_tab_pdf.table(cellText=tab_data, colLabels=tab_cols, loc='center', cellLoc='center')
                tab.auto_set_font_size(False)
                tab.set_fontsize(10)
                tab.scale(1, 1.5)
                salva_pagina_uniforme(pdf, fig_tab_pdf)

                # Pagina 3: Spettro Globale (Evidenziato se selezionato)
                fig_spec_pdf = plt.figure(dpi=300)
                ax_spec_pdf = fig_spec_pdf.add_subplot(111)
                ax_spec_pdf.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=1.5)
                if selected_delta is not None and nmr_type == '1h':
                    ax_spec_pdf.axvspan(selected_delta - width_box, selected_delta + width_box, color=BORDEAUX, alpha=0.18)
                if nmr_type == '13c' and tech in ["DEPT-135", "APT"]: ax_spec_pdf.axhline(0, color='black', linestyle='--', alpha=0.3)
                ax_spec_pdf.set_xlim(x_range[1], x_range[0])
                ax_spec_pdf.set_ylim(y_min, y_max)
                ax_spec_pdf.set_xlabel('Chemical Shift δ (ppm)', fontsize=12)
                ax_spec_pdf.set_ylabel('Intensità', fontsize=12)
                ax_spec_pdf.set_title(plot_title, fontsize=14, fontweight='bold')
                for sp in ['top', 'right']: ax_spec_pdf.spines[sp].set_visible(False)
                salva_pagina_uniforme(pdf, fig_spec_pdf)

                # Pagina 4: Zoom di tutti i multipletti
                if nmr_type == '1h' and len(segnali_visibili) > 0:
                    fig_zoom_pdf, axes = plt.subplots(1, len(segnali_visibili), dpi=300, figsize=(max(3 * len(segnali_visibili), 6), 3.5))
                    if len(segnali_visibili) == 1: axes = [axes]
                    signals_sorted = sorted(segnali_visibili, key=lambda x: float(x.get('delta', 0)), reverse=True)
                    for i, (ax, sig) in enumerate(zip(axes, signals_sorted)):
                        delta = float(sig.get('delta', 1.0))
                        ax.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=2.0) 
                        molt_f = len(sig.get('multiplicity', 's'))
                        w_zoom = 0.20 if sig.get('is_exchangeable', False) else ((0.03 * molt_f) * (500.0 / freq))
                        ax.set_xlim(delta + w_zoom, delta - w_zoom)
                        mask = (x_ppm >= delta - w_zoom) & (x_ppm <= delta + w_zoom)
                        ax.set_ylim(0, (np.max(y_intensity[mask]) if np.any(mask) else 1) * 1.1)
                        ax.set_title(f"{delta:.2f} ppm\n{sig.get('multiplicity', 's')}", fontsize=10)
                        ax.get_yaxis().set_visible(False)
                        for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
                    salva_pagina_uniforme(pdf, fig_zoom_pdf)

            st.markdown("---")
            st.download_button("Esporta Report Completo (PDF)", data=pdf_buffer.getvalue(), file_name="Report_NMR_Lab.pdf", mime="application/pdf", use_container_width=True)
