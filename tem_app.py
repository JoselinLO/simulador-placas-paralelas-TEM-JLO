import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# ======================================================================
# 1. DEFINICIÓN DE MATERIALES (CONDUCTOR Y DIELÉCTRICO)
# ======================================================================

# Materiales conductores (sigma_c, mu_r_conductor)
MATERIALES_CONDUCTORES = {
    "Aluminio": [3.82e7, 1.0], 
    "Cobre": [5.80e7, 1.0],
    "Oro": [4.10e7, 1.0],
    "Plata": [6.17e7, 1.0],
    "Hierro": [1.03e7, 500.0],
    "Níquel": [1.45e7, 100.0],
    "Latón": [1.50e7, 1.0],
    "Zinc": [1.67e7, 1.0],
    "Tungsteno": [1.82e7, 1.0]
}

# Materiales dieléctricos (tan_delta, mu_r_diel, epsilon_r)
MATERIALES_DIELECTRICOS = {
    "Aire": [0, 1.0, 1.0005],
    "Alcohol_etílico": [100.00e-3, 1.0, 25.0],
    "Oxido_de_aluminio": [0.60e-3, 1.0, 8.8],
    "Baquelita": [22.00e-3, 1.0, 4.74],
    "Dióxido_de_carbono": [0, 1.0, 1.001],
    "Vidrio": [2.00e-3, 1.0, 4.0],
    "Hielo": [50.00e-3, 1.0, 4.2],
    "Mica": [0.60e-3, 1.0, 5.4],
    "Nylon": [20.00e-3, 1.0, 3.5],
    "Papel": [8.00e-3, 1.0, 3.0],
    "Plexiglás": [30.00e-3, 1.0, 3.45],
    "Polietileno": [0.20e-3, 1.0, 2.26],
    "Polipropileno": [0.30e-3, 1.0, 2.25],
    "Poliestireno": [0.05e-3, 1.0, 2.56],
    "Porcelana": [14.00e-3, 1.0, 6.0],
    "Vidrio_Pyrex": [0.60e-3, 1.0, 4.0],
    "Cuarzo": [0.75e-3, 1.0, 3.8],
    "Hule": [2.00e-3, 1.0, 2.5],
    "Nieve": [500.00e-3, 1.0, 3.3],
    "Tierra_seca": [50.00e-3, 1.0, 2.8],
    "Teflon": [0.30e-3, 1.0, 2.1],
    "Madera_seca": [10.00e-3, 1.0, 1.5]
}


# ----------------------------------------------------------------------
# 2. FUNCIÓN DE CÁLCULO PARA LÍNEA DE PLACAS PARALELAS (TEM)
# ----------------------------------------------------------------------

def calculate_tem(f, d, W, conductor_data, dielectric_data, L):
    """
    Calcula la propagación TEM para una línea de placas paralelas, 
    incluyendo pérdidas del conductor (R) y del dieléctrico (G).
    """
    
    # Constantes Físicas
    mu0 = 4 * np.pi * 1e-7      # Permeabilidad del vacío (H/m)
    eps0 = 8.854e-12    # Permitividad del vacío (F/m)
    w = 2 * np.pi * f
    
    # Datos del conductor
    sigma_c = conductor_data[0]
    mur_c = conductor_data[1]    
    
    # Datos del dieléctrico
    tan_delta = dielectric_data[0] # Tangente de pérdidas
    er = dielectric_data[2]      # Permitividad relativa
    
    # 1. CONSTANTES RLCG POR UNIDAD DE LONGITUD
    
    # Resistencia (R) - Pérdidas en el conductor (usando Resistencia Superficial Rs)
    Rs = np.lib.scimath.sqrt((np.pi * f * mu0 * mur_c) / sigma_c)
    R = (2 * Rs) / W  # R por unidad de longitud (para ambas placas)
    
    # Inductancia (L)
    L_unit = mu0 * mur_c * d / W  # H/m (usando mu_r del conductor)
    
    # Capacitancia (C)
    C = eps0 * er * W / d  # F/m
    
    # Conductancia (G) - Pérdidas en el dieléctrico (G = w * C * tan(delta))
    G = w * C * tan_delta  # S/m
    
    # 2. CÁLCULO DE CONSTANTES DE PROPAGACIÓN
    
    # Impedancia Serie (Z) y Admitancia Paralelo (Y)
    Z = R + 1j * w * L_unit
    Y = G + 1j * w * C
    
    # Constante de Propagación (Gamma)
    gamma = np.lib.scimath.sqrt(Z * Y)
    
    # Impedancia Característica (Z0)
    Z0 = np.lib.scimath.sqrt(Z / Y)
    
    # 3. PERFILES DE V Y I
    z = np.linspace(0, L, 500) # Aumentamos los puntos para mejor visualización de la onda
    V_input = 1.0 # Tensión de entrada (1V)
    
    # Fasores complejos V(z) e I(z) para una línea adaptada
    V_z = V_input * np.exp(-gamma * z) 
    I_z = V_z / Z0
    
    return gamma, Z0, R, L_unit, C, G, z, V_z, I_z

# ----------------------------------------------------------------------
# 3. DISEÑO DE LA INTERFAZ CON STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Propagación TEM")
st.title("Simulador de Propagación de Ondas TEM (Placas Paralelas)")
st.markdown("""
    Esta herramienta calcula y visualiza las constantes de propagación ($\gamma = \\alpha + j\\beta$) y los perfiles de tensión y corriente a lo largo de una línea de transmisión de placas paralelas, considerando pérdidas del conductor y del dieléctrico.
""")

col_left, col_right = st.columns([1, 3])

# --- Columna de Configuración con estilo (col_left) ---
with col_left:
    # Contenedor para dar un toque de color y agrupar la configuración
    with st.container(border=True): 
        st.info("Configuración de la Línea")
        
        # Frecuencia
        f = st.slider("Frecuencia (Hz)", 1e6, 10e9, 6e9, format="%e") 
        
        # Control de Fase
        t_fase = st.slider("Fase de Visualización (Tiempo Angular)", 0.0, 2 * np.pi, 0.0, format="%.2f", help="Ajusta la fase angular (wt) para la gráfica de onda instantánea.")
        
        st.info("Propiedades del Material")
        
        # SELECTOR DE MATERIAL CONDUCTOR
        material_conductor_name = st.selectbox(
            "1. Placas Conductoras (Material):",
            options=list(MATERIALES_CONDUCTORES.keys()),
            key='sel_cond'
        )
        conductor_data = MATERIALES_CONDUCTORES[material_conductor_name]
        
        # SELECTOR DE MATERIAL DIELÉCTRICO
        material_diel_name = st.selectbox(
            "2. Dieléctrico (Material):",
            options=list(MATERIALES_DIELECTRICOS.keys()),
            key='sel_diel'
        )
        dielectric_data = MATERIALES_DIELECTRICOS[material_diel_name]
        
        st.info("Geometría")
        
        # Bloque SVG **Comprimido en una sola línea** para asegurar el renderizado
        svg_code = """
        <div style="text-align: center; margin: 15px 0;">
            <svg width="200" height="100" viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
                <rect x="10" y="10" width="180" height="10" style="fill: #808080; stroke: black; stroke-width: 1;"/>
                <text x="100" y="5" font-size="10" text-anchor="middle">Conductor Superior</text>
                <rect x="10" y="80" width="180" height="10" style="fill: #808080; stroke: black; stroke-width: 1;"/>
                <text x="100" y="98" font-size="10" text-anchor="middle">Conductor Inferior</text>
                <rect x="10" y="20" width="180" height="60" style="fill: #ADD8E6; opacity: 0.5;"/>
                <line x1="190" y1="10" x2="190" y2="90" style="stroke: #FF4B4B; stroke-width: 1;"/>
                <text x="195" y="50" font-size="10" fill="#FF4B4B">W (Ancho)</text>
                <line x1="5" y1="20" x2="5" y2="80" style="stroke: #FF4B4B; stroke-width: 1;"/>
                <line x1="5" y1="20" x2="10" y2="20" style="stroke: #FF4B4B; stroke-width: 1;"/>
                <line x1="5" y1="80" x2="10" y2="80" style="stroke: #FF4B4B; stroke-width: 1;"/>
                <text x="0" y="50" font-size="10" fill="#FF4B4B" text-anchor="end">d (Sep.)</text>
            </svg>
        </div>
        """
        st.markdown(svg_code, unsafe_allow_html=True)
        st.markdown("<i style='font-size: 10px; display: block; text-align: center; margin-top: -10px;'>Esquema de la sección transversal (no a escala).</i>", unsafe_allow_html=True)


        d = st.number_input("Separación entre Placas (d en m)", value=0.005, format="%e", min_value=1e-6)
        W = st.number_input("Ancho de Placas (W en m)", value=0.1, format="%e", min_value=1e-3)
        L = st.number_input("Longitud de la Línea (L en m)", value=10.0, min_value=0.001)

    st.markdown("---")
    # Botón más visual (usando la API de color de Streamlit)
    if st.button("Calcular Propagación", use_container_width=True, type="primary"):
        st.session_state['run_calc'] = True
    elif 'run_calc' not in st.session_state:
        st.session_state['run_calc'] = False

# ----------------------------------------------------------------------
# 4. EJECUCIÓN Y VISUALIZACIÓN
# ----------------------------------------------------------------------

with col_right:
    st.header("Resultados y Visualización")

    if st.session_state.get('run_calc', False):
        try:
            # Ejecutar la función
            gamma, Z0, R, L_unit, C, G, z, V_z, I_z = calculate_tem(f, d, W, conductor_data, dielectric_data, L)
            
            # --- DATOS PARA GRÁFICAS DE ONDA COMPLETA (NO ABSOLUTO) ---
            V_temporal = np.real(V_z * np.exp(1j * t_fase))
            I_temporal = np.real(I_z * np.exp(1j * t_fase))
            V_mag = np.abs(V_z)
            I_mag = np.abs(I_z)
            
            # ------------------------------------------------------------------
            # DETALLE DE PROPIEDADES Y RESULTADOS
            # ------------------------------------------------------------------
            
            # Propiedades utilizadas
            with st.expander("Propiedades Físicas Utilizadas", expanded=False):
                col_prop1, col_prop2 = st.columns(2)
                
                # Propiedades del Conductor
                col_prop1.markdown(f"**Conductor:** `{material_conductor_name}`")
                col_prop1.markdown(f"$\sigma_c$: **{conductor_data[0]:.2e}** S/m")
                col_prop1.markdown(f"$\mu_r$: **{conductor_data[1]}**")

                # Propiedades del Dieléctrico
                col_prop2.markdown(f"**Dieléctrico:** `{material_diel_name}`")
                col_prop2.markdown(f"$\epsilon_r$: **{dielectric_data[2]:.3f}**")
                # Se mantiene la notación LaTeX para evitar errores de traducción
                col_prop2.markdown(f"$$\\tan\\delta$$: **{dielectric_data[0]:.2e}**")


            # Constantes de Propagación
            st.info("Constantes de Propagación")
            alpha = np.real(gamma)
            beta = np.imag(gamma)
            
            # Usando st.metric para un look más moderno y colorido
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Atenuación ($\alpha$)", value=f"{alpha:.4e} Np/m", help="Pérdida de amplitud a lo largo de la línea.")
            col2.metric(label="Fase ($\beta$)", value=f"{beta:.4e} rad/m", help="Cambio de fase por unidad de longitud.")
            col3.metric(label="Impedancia Característica ($|Z_0|$)", value=f"{np.abs(Z0):.2f} Ω", help="Relación entre la tensión y la corriente de la onda viajera.")
            
            # Constantes RLCG
            with st.expander("Constantes RLCG por Unidad de Longitud", expanded=False):
                col_rl, col_cg = st.columns(2)
                col_rl.metric(label="Resistencia (R)", value=f"{R:.4e} Ω/m")
                col_rl.metric(label="Inductancia (L)", value=f"{L_unit:.4e} H/m")
                col_cg.metric(label="Capacitancia (C)", value=f"{C:.4e} F/m")
                col_cg.metric(label="Conductancia (G)", value=f"{G:.4e} S/m")

            st.markdown("---") 

            st.info("Perfiles de Tensión y Corriente")

            # Figura 1: Tensión V(z)
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            
            # 1. Gráfica de la onda instantánea (parte real del fasor)
            ax1.plot(z, V_temporal, color='#1f77b4', label=r'$v(z, t_0)$', linewidth=2, alpha=0.8) # Azul
            
            # 2. Gráfica de la envolvente de atenuación (|V(z)|)
            ax1.plot(z, V_mag, 'r--', label='$|V(z)|$', linewidth=1.5, alpha=0.6) # Rojo claro para envolvente
            ax1.plot(z, -V_mag, 'r--', linewidth=1.5, alpha=0.6) # Envolvente negativa
            
            ax1.axhline(0, color='black', linestyle='-', linewidth=0.5) # Eje de las abscisas más oscuro
            ax1.set_title(f"Perfil de Tensión - Onda Instantánea y Atenuación")
            ax1.set_xlabel("Posición z (m)")
            ax1.set_ylabel("Tensión (V)")
            ax1.legend()
            ax1.grid(True, linestyle=':', alpha=0.6)
            y_limit = np.max(V_mag) * 1.05
            ax1.set_ylim(-y_limit, y_limit)
            st.pyplot(fig1)

            # Figura 2: Corriente I(z)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            
            # 1. Gráfica de la onda instantánea (parte real del fasor)
            ax2.plot(z, I_temporal, color='#2ca02c', label=r'$i(z, t_0)$', linewidth=2, alpha=0.8) # Verde
            
            # 2. Gráfica de la envolvente de atenuación (|I(z)|)
            ax2.plot(z, I_mag, color='#9467bd', linestyle='--', label='$|I(z)|$', linewidth=1.5, alpha=0.6) # Púrpura claro para envolvente
            ax2.plot(z, -I_mag, color='#9467bd', linestyle='--', linewidth=1.5, alpha=0.6) # Envolvente negativa
            
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5) # Eje de las abscisas más oscuro
            ax2.set_title(f"Perfil de Corriente - Onda Instantánea y Atenuación")
            ax2.set_xlabel("Posición z (m)")
            ax2.set_ylabel("Corriente (A)")
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.6)
            y_limit = np.max(I_mag) * 1.05
            ax2.set_ylim(-y_limit, y_limit)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
            st.error("Asegúrate de que los parámetros de entrada sean válidos.")
            st.info("Nota: Para frecuencias y pérdidas muy altas, las funciones como np.lib.scimath.sqrt manejan correctamente los números complejos.")

    else:
        st.info("Presiona 'Calcular Propagación' para iniciar la simulación y ver los resultados.")