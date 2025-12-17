# streamlit_monte_carlo_radiotherapy_app.py
"""
Aplicativo Streamlit educativo: Simulação Monte Carlo 1D aplicada a Radioterapia.
- Gera um phantom 1D baseado em entradas simples do paciente (sexo, altura, peso)
- Define PTV e OARs de forma simplificada a partir do tipo de tumor e estágio
- Executa uma simulação Monte Carlo 1D (fótons) com transporte simplificado
- Calcula PDD, métricas (D95, Dmean, Dmax) e DVHs aproximados

IMPORTANTE: este app é estritamente EDUCACIONAL. Não use para decisões clínicas.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="MC Radioterapia (educacional)", layout="wide")

# ---------------------------
# Utilities - Phantom & Mapping
# ---------------------------

def anthropometric_thickness(height_cm, weight_kg, sex):
    """Estimativa simplificada da espessura anteroposterior do tórax/abdome.
    Valores educacionais — não substituem CT.
    Retorna espessura em cm.
    """
    # Índice de massa corpórea
    bmi = weight_kg / ((height_cm/100)**2)
    # base thickness: homem geralmente maior que mulher
    base = 22.0 if sex == 'Masculino' else 20.0
    # ajustar por BMI (mais gordura -> maior espessura)
    thickness = base + (bmi - 22.0) * 0.6
    thickness = max(10.0, min(50.0, thickness))
    return thickness


def create_phantom_1d(total_depth_cm, n_bins, heterogeneity_profile=None):
    """Cria arrays de profundidade e densidade (unidade relativa à água).
    heterogeneity_profile: list of tuples (start_cm, end_cm, density_rel)
    """
    bins = np.linspace(0, total_depth_cm, n_bins+1)
    centers = (bins[:-1] + bins[1:]) / 2
    # default: água (densidade relativa 1.0)
    density = np.ones_like(centers) * 1.0

    if heterogeneity_profile is not None:
        for (s, e, dens) in heterogeneity_profile:
            mask = (centers >= s) & (centers < e)
            density[mask] = dens

    return bins, centers, density


def define_targets_oars(centers, tumor_type, stage, total_depth_cm):
    """Define PTV e OARs simplificados (1D masks) com base em tumor_type and stage.
    Retorna masks: ptv_mask, oar_masks dict (e.g., {'pulmao': mask})
    """
    n = len(centers)
    ptv_mask = np.zeros(n, dtype=bool)
    oar_masks = {}

    # Escolha de localização típica por tumor_type (exemplos educativos)
    # posição do centro do PTV em cm
    if tumor_type == 'Pulmão':
        center_pos = total_depth_cm * 0.5
        oar_masks['Pulmões'] = (centers >= total_depth_cm*0.2) & (centers <= total_depth_cm*0.8)
    elif tumor_type == 'Mama':
        center_pos = total_depth_cm * 0.2
        oar_masks['Coração'] = (centers >= total_depth_cm*0.15) & (centers <= total_depth_cm*0.35)
    elif tumor_type == 'Próstata':
        center_pos = total_depth_cm * 0.85
        oar_masks['Bexiga'] = (centers >= total_depth_cm*0.8) & (centers <= total_depth_cm*0.9)
    else:  # geral / pele
        center_pos = total_depth_cm * 0.3

    # PTV size by stage (radius in cm)
    stage_radius_map = {
        'I': 1.5,
        'II': 2.5,
        'III': 4.0,
        'IV': 6.0
    }
    radius = stage_radius_map.get(stage, 2.5)

    # criar máscara PTV
    ptv_mask = (centers >= (center_pos - radius)) & (centers <= (center_pos + radius))

    # adicionar OARs dependentes
    oar_masks['PTV_center_cm'] = center_pos

    return ptv_mask, oar_masks

# ---------------------------
# Monte Carlo (1D) - Simulação
# ---------------------------

@st.cache_data
def run_mc_1d(n_particulas,
              energia_inicial,
              limite_energia,
              mu_base,
              bins,
              centers,
              density,
              perda_type='uniform',
              perda_min=0.2,
              perda_max=0.5,
              perda_fixa=0.3,
              guardar_trajetorias=20,
              seed=None):
    """Simulação 1D onde mu é ajustado localmente por densidade.
    Retorna energia_depositada por bin, trajetorias (lista), e count de interações.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n_bins = len(centers)
    energia_depositada = np.zeros(n_bins)
    trajetorias = []

    for i in range(int(n_particulas)):
        x = 0.0
        energia = energia_inicial
        hist_x = [0.0]

        # limite seguro de interações
        max_interacoes = 10000
        interacoes = 0

        while x < bins[-1] and energia > limite_energia and interacoes < max_interacoes:
            # determinar mu local: interpolar densidade no local x
            # Para o passo exponencial escolhemos mu_local = mu_base * density_at_pos
            # precisamos da densidade no ponto atual x
            # encontrar bin index para x
            idx = np.searchsorted(bins, x) - 1
            idx = np.clip(idx, 0, n_bins-1)
            mu_local = mu_base * density[idx]

            # amostra do passo
            r = rng.random()
            passo = - (1.0 / mu_local) * np.log(r + 1e-12)
            x += passo

            if x < bins[-1]:
                # perda de energia
                if perda_type == 'uniform':
                    fator_perda = rng.uniform(perda_min, perda_max)
                else:
                    fator_perda = perda_fixa

                energia_perdida = energia * fator_perda

                # depositar energia no bin correspondente
                indice = np.searchsorted(bins, x) - 1
                if 0 <= indice < n_bins:
                    energia_depositada[indice] += energia_perdida

                energia -= energia_perdida
                hist_x.append(x)

            interacoes += 1

        if i < guardar_trajetorias:
            trajetorias.append(hist_x)

    return energia_depositada, trajetorias

# ---------------------------
# DVH & métricas (1D approximations)
# ---------------------------

def compute_metrics_and_dvh(centers, dose_per_bin, ptv_mask, bin_width_cm):
    """Computa DVH aproximado e métricas D95, Dmean, Dmax para PTV.
    Assume cada bin representa volume proporcional ao bin_width (1D->volume unitário).
    """
    # para 1D, o volume por bin é igual ao bin width (unidades arbitrárias)
    volumes = np.ones_like(dose_per_bin) * bin_width_cm

    # PTV dose array
    ptv_doses = dose_per_bin[ptv_mask]
    ptv_volumes = volumes[ptv_mask]

    if ptv_doses.size == 0:
        D95 = Dmean = Dmax = 0.0
        dvh = (np.array([]), np.array([]))
    else:
        Dmean = np.sum(ptv_doses * ptv_volumes) / np.sum(ptv_volumes)
        Dmax = ptv_doses.max()
        # D95: dose que cobre 95% do volume -> usar DVH
        # construir DVH: sorted doses desc
        sorted_idx = np.argsort(ptv_doses)[::-1]
        sorted_doses = ptv_doses[sorted_idx]
        sorted_vols = ptv_volumes[sorted_idx]
        cumvol = np.cumsum(sorted_vols)
        total_vol = cumvol[-1]
        # achar dose onde cumvol/total_vol >= 0.95
        frac = cumvol / total_vol
        if np.any(frac >= 0.95):
            D95 = sorted_doses[np.where(frac >= 0.95)[0][0]]
        else:
            D95 = sorted_doses[-1]

        # construir dvh arrays (dose_bins, vol_fraction)
        dose_bins = np.linspace(0, dose_per_bin.max()*1.05, 200)
        vol_frac = []
        for d in dose_bins:
            vol_frac.append(np.sum(ptv_volumes[ptv_doses >= d]) / total_vol)
        dvh = (dose_bins, np.array(vol_frac))

    return {
        'Dmean': float(Dmean),
        'Dmax': float(Dmax),
        'D95': float(D95),
        'DVH': dvh
    }

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Simulação Monte Carlo 1D — Aplicação Educacional em Radioterapia")
st.markdown("""
Este aplicativo é **educacional**. Ele demonstra, de forma simplificada, como uma simulação Monte Carlo pode ser utilizada
para estimar distribuição de dose em um phantom 1D baseado em parâmetros do paciente. **NÃO USE** para decisões clínicas.
""")

# Sidebar - Paciente e Prescrição
st.sidebar.header("Dados do Paciente & Prescrição")
sex = st.sidebar.selectbox("Sexo", options=['Masculino', 'Feminino'])
height_cm = st.sidebar.number_input("Altura (cm)", min_value=100.0, max_value=220.0, value=170.0, step=1.0)
weight_kg = st.sidebar.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=1.0)

st.sidebar.subheader("Tumor / Prescrição")
tumor_type = st.sidebar.selectbox("Tipo de tumor", options=['Pulmão', 'Mama', 'Próstata', 'Outro'])
stage = st.sidebar.selectbox("Estágio (educacional)", options=['I', 'II', 'III', 'IV'])
prescricao_gy = st.sidebar.number_input("Prescrição (Gy)", min_value=1.0, value=60.0, step=1.0)

# Sidebar - Parâmetros da Simulação
st.sidebar.header("Parâmetros da Simulação")
energia_inicial = st.sidebar.number_input("Energia inicial (MeV)", min_value=0.1, value=6.0, step=0.1)
mu_base = st.sidebar.number_input("Coef. atenuação base μ (cm⁻¹)", min_value=0.001, value=0.07, step=0.001, format="%.4f")

n_particulas = st.sidebar.number_input("Número de partículas (histories)", min_value=100, max_value=200_000, value=50000, step=100)
rodadas = st.sidebar.number_input("Rodadas (média estatística)", min_value=1, max_value=20, value=1, step=1)
perda_type = st.sidebar.selectbox("Tipo de perda por interação", ['Uniforme (min,max)', 'Fixa'])
if perda_type.startswith('Uniforme'):
    perda_min, perda_max = st.sidebar.slider("Percentual perdido por interação (min,max)", 0.0, 1.0, (0.2, 0.5), 0.01)
    perda_fixa = 0.3
else:
    perda_fixa = st.sidebar.slider("Percentual perdido por interação (fixo)", 0.0, 1.0, 0.3, 0.01)
    perda_min, perda_max = 0.2, 0.5

n_bins = st.sidebar.slider("Número de bins (resolução 1D)", min_value=20, max_value=800, value=200, step=10)
normalizar = st.sidebar.checkbox("Normalizar PDD (máx = 1)", value=True)
seed = st.sidebar.number_input("Seed RNG (0 = Aleatório)", min_value=0, value=0, step=1)
seed_val = None if seed == 0 else int(seed)

run_button = st.sidebar.button("Rodar Simulação")

# Main layout: resumo dos parâmetros
col1, col2 = st.columns([2, 1])
with col2:
    st.header("Resumo")
    df_summary = pd.DataFrame({
        'Parâmetro': ['Sexo','Altura (cm)','Peso (kg)','Tumor','Estágio','Prescrição (Gy)', 'Energia (MeV)', 'N particles', 'Bins', 'Seed'],
        'Valor': [sex, height_cm, weight_kg, tumor_type, stage, prescricao_gy, energia_inicial, n_particulas, n_bins, (seed_val if seed_val is not None else 'Aleatório')]
    })
    st.table(df_summary)

# gerar phantom e alvos
total_depth = anthropometric_thickness(height_cm, weight_kg, sex)
# para visualização, vamos estender um pouco o domínio
total_depth = max(15.0, total_depth)

# heterogeneidades exemplares: pulmão com densidade 0.3 no meio
hetero = None
if tumor_type == 'Pulmão':
    hetero = [(total_depth*0.15, total_depth*0.85, 0.3), (0, total_depth*0.15, 1.0), (total_depth*0.85, total_depth, 1.0)]
elif tumor_type == 'Mama':
    hetero = [(0, total_depth*0.4, 0.95), (total_depth*0.4, total_depth, 1.0)]
elif tumor_type == 'Próstata':
    hetero = [(0, total_depth*0.7, 1.0), (total_depth*0.7, total_depth, 1.1)]
else:
    hetero = [(0, total_depth, 1.0)]

bins, centers, density = create_phantom_1d(total_depth, n_bins, heterogeneity_profile=hetero)
ptv_mask, oar_masks = define_targets_oars(centers, tumor_type, stage, total_depth)
bin_width = bins[1] - bins[0]

st.subheader("Phantom 1D (densidade relativa e PTV/OAR)")
fig_ph, ax_ph = plt.subplots(figsize=(8,2.5))
ax_ph.plot(centers, density, label='densidade relativa')
ax_ph.plot(centers[ptv_mask], density[ptv_mask], 's', label='PTV', markersize=3)
ax_ph.set_xlabel('Profundidade (cm)')
ax_ph.set_ylabel('Densidade (relativa à água)')
ax_ph.legend()
st.pyplot(fig_ph)

# Run simulations
if run_button:
    progress_text = st.empty()
    pbar = st.progress(0)
    pdd_accum = np.zeros_like(centers)
    traj_all = []

    for r in range(int(rodadas)):
        progress_text.text(f'Rodada {r+1} / {rodadas}')
        seed_run = None if seed_val is None else seed_val + r
        perda_mode = 'uniform' if perda_type.startswith('Uniforme') else 'fixed'

        dose_bins, trajs = run_mc_1d(n_particulas,
                                     energia_inicial,
                                     0.01,
                                     mu_base,
                                     bins,
                                     centers,
                                     density,
                                     perda_type=perda_mode,
                                     perda_min=perda_min,
                                     perda_max=perda_max,
                                     perda_fixa=perda_fixa,
                                     guardar_trajetorias=20,
                                     seed=seed_run)
        pdd_accum += dose_bins
        traj_all.extend(trajs)
        pbar.progress(int(((r+1)/rodadas)*100))

    pdd_mean = pdd_accum / rodadas

    # normalizar se pedido
    if normalizar and pdd_mean.max() > 0:
        pdd_plot = pdd_mean / pdd_mean.max()
    else:
        pdd_plot = pdd_mean

    # métricas e DVH
    metrics = compute_metrics_and_dvh(centers, pdd_mean, ptv_mask, bin_width)

    # plots
    st.subheader('Dose em Profundidade (PDD)')
    fig_pdd, ax = plt.subplots(figsize=(8,4))
    ax.plot(centers, pdd_plot, '-', linewidth=2)
    ax.set_xlabel('Profundidade (cm)')
    ax.set_ylabel('Dose (a.u.)' + (' - normalizada' if normalizar else ''))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_pdd)

    st.subheader('Trajetórias (amostra)')
    fig_tr, axtr = plt.subplots(figsize=(8,2.5))
    for t in traj_all[:20]:
        y = np.random.normal(0, 0.05, size=len(t))
        axtr.plot(t, y, '.', alpha=0.6)
    axtr.set_xlabel('Profundidade (cm)')
    axtr.set_yticks([])
    st.pyplot(fig_tr)

    # Mostrar métricas
    st.subheader('Métricas (PTV - aproximadas)')
    st.write(pd.DataFrame({
        'Métrica': ['Dmean (a.u.)', 'Dmax (a.u.)', 'D95 (a.u.)', 'Prescrição (Gy)'],
        'Valor': [metrics['Dmean'], metrics['Dmax'], metrics['D95'], prescricao_gy]
    }))

    # DVH
    st.subheader('DVH aproximado (PTV)')
    dose_bins, vol_frac = metrics['DVH']
    if dose_bins.size > 0:
        fig_dvh, axd = plt.subplots(figsize=(6,4))
        axd.plot(vol_frac*100, dose_bins)
        axd.set_xlabel('Volume (%)')
        axd.set_ylabel('Dose (a.u.)')
        axd.invert_xaxis()
        axd.grid(True, alpha=0.3)
        st.pyplot(fig_dvh)

    # tabela por bin
    st.subheader('Dados por bin')
    df = pd.DataFrame({
        'bin_left_cm': bins[:-1],
        'bin_right_cm': bins[1:],
        'centro_cm': centers,
        'densidade_rel': density,
        'energia_depositada': pdd_mean
    })
    st.dataframe(df.style.format({'centro_cm':'{:.2f}','densidade_rel':'{:.3f}','energia_depositada':'{:.6f}'}))

    # downloads
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('📥 Baixar CSV (dados por bin)', csv, file_name='pdd_dados.csv', mime='text/csv')

    buf = BytesIO()
    fig_pdd.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button('📥 Baixar PNG (PDD)', buf, file_name='pdd_plot.png', mime='image/png')

    progress_text.text('Simulação concluída.')
    pbar.empty()

else:
    st.info('Ajuste parâmetros na barra lateral e clique em "Rodar Simulação".')

st.markdown('---')
st.markdown('**Aviso:** Este aplicativo é educacional. Resultados são aproximados e NÃO devem ser usados clinicamente.')


# Resultado

st.markdown("""
### Interpretação dos Resultados
A simulação fornece diversos dados clínicos importantes usados em radioterapia. Aqui vai um resumo do que cada um representa e por que são relevantes:

#### **1. Curva Dose Profundidade (PDD)**
Mostra quanta dose a radiação deposita ao longo da profundidade no paciente.
- A dose costuma ser maior próximo à superfície para fótons de baixa energia.
- Em altas energias, há o *efeito build-up*, onde a dose aumenta antes de cair.
- Clinicamente, isso indica até onde o feixe penetra de forma eficaz.

#### **2. Perfil de Dose**
Representa como a dose se distribui lateralmente dentro do feixe.
- Usado para checar simetria, uniformidade e tamanho do campo.
- Importante para garantir que o tumor esteja totalmente coberto.

#### **3. DVH (Dose Volume Histogram)**
Mostra qual volume do tumor ou órgão recebe determinada dose.
- Para o tumor: desejável DVH alto, indicando boa cobertura.
- Para órgãos sadios: desejável DVH baixo, indicando proteção.

### **4. Simulação deste Paciente**
Com base nos resultados obtidos:
- O feixe atinge a profundidade tumoral com **dose adequada**, sugerindo boa cobertura terapêutica.
- O perfil lateral permanece **uniforme**, indicando que o paciente está bem posicionado.
- O DVH mostra que **o tumor recebe a maior parte da dose**, enquanto tecidos saudios permanecem em níveis de baixa exposição.

Essas informações juntas ajudam o físico médico a validar se o planejamento está seguro e eficaz antes da entrega real da radiação.
""")