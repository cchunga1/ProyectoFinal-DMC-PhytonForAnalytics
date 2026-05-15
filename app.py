import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la página
st.set_page_config(page_title="Análisis Exploratorio de Datos (EDA) - Telco Customer Chum", layout="wide")

# ==========================================
# PROGRAMACIÓN ORIENTADA A OBJETOS (POO)
# ==========================================
class DataAnalyzer:
    """Clase para encapsular el análisis y visualización de datos."""
    def __init__(self, dataframe):
        # df_data: Variable tipo DataFrame que almacena los datos tabulares del CSV cargado
        self.df_data = dataframe
   
    # Esta funcion clasificar_variables es para el Item 2 dentro del Módulo 3 : Análisis de EDA - cc
    def clasificar_variables(self):
        """Retorna columnas numéricas y categóricas."""
        # lst_num_cols: Lista de cadenas de texto con los nombres de las columnas numéricas (int, float)
        lst_num_cols = self.df_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # lst_cat_cols: Lista de cadenas de texto con los nombres de las columnas categóricas (object, category)
        lst_cat_cols = self.df_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return lst_num_cols, lst_cat_cols

    def obtener_estadisticas(self):
        """Retorna estadísticas descriptivas del dataset."""
        # df_stats: DataFrame que contiene el resumen de las métricas estadísticas . Dataframe ya me trae la media , mediana y dispersión. Otra opcion era una usar statistics.mean(datos), statistics.median(datos), statistics.mode(datos)
        df_stats = self.df_data.describe()
        return df_stats

    def plot_histograma(self, columna, bins):
        """Genera un histograma para una columna numérica."""
        # fig_hist: Objeto de figura de Matplotlib que contendrá el gráfico del histograma
        # ax_hist: Eje individual de Matplotlib donde se dibujará la distribución
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df_data[columna], bins=bins, kde=True, ax=ax_hist)
        ax_hist.set_title(f'Distribución de {columna}')
        return fig_hist

    def plot_barras(self, columna):
        """Genera un gráfico de barras para una columna categórica."""
        # fig_bar: Objeto de figura de Matplotlib para visualizar proporciones categóricas
        # ax_bar: Eje individual de Matplotlib para las barras
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        sns.countplot(data=self.df_data, x=columna, ax=ax_bar, palette="viridis")
        ax_bar.set_title(f'Conteo de {columna}')
        plt.xticks(rotation=45)
        return fig_bar
    
    def plot_bivariado_num_cat(self, col_num, col_cat):
        """Genera un boxplot para analizar variable numérica vs categórica."""
        # fig_box: Objeto de figura de Matplotlib utilizado para dibujar gráficos de cajas (boxplots)
        # ax_box: Eje individual de Matplotlib para la dispersión y cuartiles
        fig_box, ax_box = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.df_data, x=col_cat, y=col_num, ax=ax_box, palette="Set2")
        ax_box.set_title(f'{col_num} vs {col_cat}')
        return fig_box

# ==========================================
# INTERFAZ PRINCIPAL Y NAVEGACIÓN
# ==========================================
def main():
    # str_menu_option: Cadena de texto que indica la pestaña seleccionada por el usuario en la barra lateral
    str_menu_option = st.sidebar.radio(
        "Menú de Navegación principal",
        ["Módulo 1: Home", "Módulo 2: Carga del dataset", "Módulo 3: Análisis (EDA)", "Módulo 4: Conclusiones"]
    )

    if str_menu_option == "Módulo 1: Home":
        mostrar_home()
    elif str_menu_option == "Módulo 2: Carga del dataset":
        cargar_datos()
    elif str_menu_option == "Módulo 3: Análisis (EDA)":
        mostrar_eda()
    elif str_menu_option == "Módulo 4: Conclusiones":
        mostrar_conclusiones()

# ==========================================
# MÓDULO 1: HOME
# ==========================================
def mostrar_home():
    st.title("Proyecto: Análisis Exploratorio de Datos (EDA) - Telco Customer Churn")
    st.markdown("### Objetivo del Análisis")
    st.info("El objetivo es analizar, limpiar, transformar y visualizar los datos históricos para identificar patrones asociados a la fuga (churn) de clientes, utilizando un enfoque exploratorio y visual.")
    
    # st.markdown("### Datos del Autor")
    # st.write("**Nombre:** [Ingresa aquí tu Nombre Completo]")
    # st.write("**Especialización:** Python for Analytics")
    # st.write("**Año:** 2026")

    st.markdown("""
    * **Nombre:** JOSE CHRISTIAN CHUNGA MARTINEZ
    * **Nombre del Módulo :** Módulo 2 - Python For Analytics | Especialización for Analytics
    * **Información general del estudiante :** Ingeniero de Sistemas
    * **Año:** 2026
    * **Tecnologías utilizadas:** Streamlit , Pandas , Numpy , Matplotlib , Seaborn
    * **URL del Proyecto : https://proyectofinal-dmc-phytonforanalytics-cc.streamlit.app/
    
    **Descripción del Proyecto:**
    Esta plataforma simula una aplicación interactiva construida en Python usando Streamlit orientada al Análisis Exploratorio de Datos(EDA)
    
    """)
    
    st.markdown("### Sobre el Dataset")
    st.write("El dataset 'TelcoCustomerChurn.csv' contiene información sobre los clientes de una empresa de telecomunicaciones, los servicios contratados, facturación mensual, tiempo de permanencia y su estado actual en la empresa.")
    
    st.markdown("### Tecnologías Utilizadas")
    st.write("- Python, Pandas, NumPy")
    st.write("- Matplotlib, Seaborn")
    st.write("- Streamlit (Interfaz interactiva)")

# ==========================================
# MÓDULO 2: CARGA DEL DATASET
# ==========================================
def cargar_datos():
    st.title("Carga del Dataset")
    st.write("Antes de ejecutar el análisis exploratorio, por favor carga el conjunto de datos.")
    
    # file_csv: Variable de tipo 'UploadedFile' que contiene temporalmente el archivo subido al sistema
    file_csv = st.file_uploader("Sube el archivo TelcoCustomerChurn.csv", type=["csv"])
    
    if file_csv is not None:
        try:
            # df_telecom: DataFrame principal que almacena en memoria los datos extraídos del CSV
            df_telecom = pd.read_csv(file_csv)
            # Guardar en estado de sesión para que los datos persistan al cambiar de módulo
            st.session_state['df_data'] = df_telecom
            
            st.success("¡Archivo cargado correctamente!")
            
            st.subheader("Vista previa de los datos (Head)")
            st.dataframe(df_telecom.head())
            
            st.subheader("Dimensiones del dataset")
            
            # num_filas: Valor entero con la cantidad de observaciones (clientes) del conjunto de datos
            # num_columnas: Valor entero con la cantidad de variables/atributos (características)
            num_filas, num_columnas = df_telecom.shape
            
            # f-strings utilizadas para dar formato a la salida
            st.write(f"**Filas totales:** {num_filas}")
            st.write(f"**Columnas totales:** {num_columnas}")
            
        except Exception as e:
            # str_error: Cadena de texto con la excepción en caso de que falle la lectura de pandas
            str_error = str(e)
            st.error(f"Error al leer el archivo: {str_error}")
    else:
        st.warning("Por favor, sube el archivo CSV para habilitar el Módulo 3.")

# ==========================================
# MÓDULO 3: EDA
# ==========================================
def mostrar_eda():
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    # Validación exigida: Ningún análisis debe ejecutarse si no hay datos
    if 'df_data' not in st.session_state:
        st.error("Error: Ningún análisis puede ejecutarse si el archivo no ha sido cargado. Ve al Módulo 2 primero.")
        return
    
    # df_actual: DataFrame que se recupera del caché/sesión de Streamlit para el análisis continuo
    df_actual = st.session_state['df_data']
    
    # Pre-procesamiento de limpieza básico (A menudo la columna TotalCharges tiene espacios en blanco)
    df_actual['TotalCharges'] = pd.to_numeric(df_actual['TotalCharges'], errors='coerce')
    
    # obj_analyzer: Instancia activa de la clase DataAnalyzer para llamar a los métodos orientados a objetos
    obj_analyzer = DataAnalyzer(df_actual)
    
    # Creación de 10 pestañas (tabs) para estructurar el análisis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 , tab11 = st.tabs([
        "1. Info General", "2. Clasificación", "3. Estadísticas", "4. Nulos", 
        "5. Dist. Numéricas", "6. Dist. Categóricas", "7. Num vs Cat", 
        "8. Cat vs Cat", "9. Dinámico", "10. Hallazgos", "11. Diccionario de datos"
    ])

    with tab1:
        st.header("Ítem 1: Información general del dataset")
        
        # df_info_custom: DataFrame creado manualmente para reemplazar el print de .info()
        df_info_custom = pd.DataFrame({
            'Tipo de Dato': df_actual.dtypes,
            'Valores No Nulos': df_actual.notnull().sum(),
            'Valores Nulos': df_actual.isnull().sum()
        })
        
        st.write("**Estructura de tipos de datos:**")
        st.dataframe(df_info_custom)
        
        # num_total_nulos: Sumatoria global de nulos
        num_total_nulos = df_info_custom['Valores Nulos'].sum()
        st.write(f"**Conteo total global de valores nulos:** {num_total_nulos}")

    with tab2:
        st.header("Ítem 2: Clasificación de variables")
        # lst_num, lst_cat: Dos listas de cadenas obtenidas al invocar la función de la clase DataAnalyzer
        lst_num, lst_cat = obj_analyzer.clasificar_variables()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Variables Numéricas ({len(lst_num)})")
            st.write(lst_num)
        with col2:
            st.subheader(f"Variables Categóricas ({len(lst_cat)})")
            st.write(lst_cat)

    with tab3:
        st.header("Ítem 3: Estadísticas descriptivas")
        st.write("Uso de la función `.describe()` para variables continuas.")
        # df_desc: DataFrame resultante del método de la clase que retorna los percentiles, media y conteo
        df_desc = obj_analyzer.obtener_estadisticas()
        st.dataframe(df_desc)
        st.info("Interpretación: Podemos comparar la media ('mean') con la mediana ('50%'). Diferencias considerables podrían indicar que los datos están sesgados (outliers).")

    with tab4:
        st.header("Ítem 4: Análisis de valores faltantes")
        # df_nulos_conteo: Series de Pandas convertida en listado de aquellas columnas que superan 0 nulos
        df_nulos_conteo = df_actual.isnull().sum()
        df_nulos_conteo = df_nulos_conteo[df_nulos_conteo > 0]
        
        if len(df_nulos_conteo) > 0:
            st.write("Conteo de nulos detectados por variable:")
            st.dataframe(df_nulos_conteo)
            
            # fig_nulos, ax_nulos: Elementos del gráfico de barras para visualizar la cantidad de vacíos
            fig_nulos, ax_nulos = plt.subplots(figsize=(6, 3))
            sns.barplot(x=df_nulos_conteo.index, y=df_nulos_conteo.values, ax=ax_nulos, palette="Reds")
            ax_nulos.set_title("Variables con Datos Faltantes")
            st.pyplot(fig_nulos)
            
            st.write("**Discusión:** Los valores nulos (típicamente en TotalCharges) surgieron de cuentas con 0 meses de antigüedad (tenure). Recomendamos imputar estos valores con 0 o eliminarlos.")
        else:
            st.success("No se encontraron variables con valores nulos pendientes.")

    with tab5:
        st.header("Ítem 5: Distribución de variables numéricas")
        lst_num, _ = obj_analyzer.clasificar_variables()
        
        # str_var_num: Variable de tipo texto donde el usuario selecciona la columna a graficar
        str_var_num = st.selectbox("Selecciona una métrica numérica:", lst_num, key="num_dist")
        
        # num_bins_slider: Valor entero capturado por el slider del usuario para el nivel de detalle del histograma
        num_bins_slider = st.slider("Ajustar detalle (bins)", min_value=10, max_value=100, value=30)
        
        # fig_hist: Gráfico retornado por la instancia de DataAnalyzer
        fig_hist = obj_analyzer.plot_histograma(str_var_num, num_bins_slider)
        st.pyplot(fig_hist)

    with tab6:
        st.header("Ítem 6: Análisis de variables categóricas")
        _, lst_cat = obj_analyzer.clasificar_variables()
        # Filtramos 'customerID' ya que tiene demasiados valores únicos para un gráfico de barras
        if 'customerID' in lst_cat: 
            lst_cat.remove('customerID')
        
        # str_var_cat: Variable de tipo texto de la selección categórica
        str_var_cat = st.selectbox("Selecciona un segmento categórico:", lst_cat, key="cat_dist")
        
        col1, col2 = st.columns(2)
        with col1:
            # fig_barras: Gráfico devuelto por DataAnalyzer con el conteo 
            fig_barras = obj_analyzer.plot_barras(str_var_cat)
            st.pyplot(fig_barras)
        with col2:
            st.write("**Proporciones (%) de cada categoría:**")
            # df_props: Series con los porcentajes relativos de la categoría
            df_props = df_actual[str_var_cat].value_counts(normalize=True) * 100
            st.dataframe(df_props.round(2).astype(str) + ' %')

    with tab7:
        st.header("Ítem 7: Análisis bivariado (Numérico vs Categórico)")
        st.write("Evalúa cómo varían las métricas de dinero/tiempo dependiendo de un grupo de clientes.")
        
        col1, col2 = st.columns(2)
        with col1:
            # str_biv_num: Selección de la variable numérica (Eje Y)
            str_biv_num = st.selectbox("Métrica (Numérica Y):", lst_num, key="biv_n")
        with col2:
            # str_biv_cat: Selección del grupo (Eje X)
            str_biv_cat = st.selectbox("Agrupación (Categórica X):", ["Churn", "Contract", "gender", "InternetService"], key="biv_c")
            
        # fig_cajas: Figura de Boxplot generada
        fig_cajas = obj_analyzer.plot_bivariado_num_cat(str_biv_num, str_biv_cat)
        st.pyplot(fig_cajas)

    with tab8:
        st.header("Ítem 8: Análisis bivariado (Categórico vs Categórico)")
        st.write("Analicemos cómo impactan los distintos servicios sobre la tasa de Fuga (Churn).")
        
        # str_servicio_cat: Servicio categórico a cruzar contra Churn
        str_servicio_cat = st.selectbox("Elige un servicio para cruzar con Churn:", ["Contract", "InternetService", "PaymentMethod", "TechSupport"])
        
        # fig_doble, ax_doble: Componentes para el gráfico de barras agrupadas de Matplotlib
        fig_doble, ax_doble = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df_actual, x=str_servicio_cat, hue="Churn", ax=ax_doble, palette="muted")
        ax_doble.set_title(f"Relación entre {str_servicio_cat} y Churn")
        st.pyplot(fig_doble)

    with tab9:
        st.header("Ítem 9: Análisis basado en parámetros seleccionados")
        # bool_usar_filtro: Bandera lógica obtenida de un checkbox para habilitar el filtrado de la base de datos
        bool_usar_filtro = st.checkbox("Habilitar filtrado por Tipo de Contrato", value=False)
        
        # df_dinamico: Sub-DataFrame temporal utilizado para mostrar datos restringidos
        df_dinamico = df_actual
        if bool_usar_filtro:
            # lst_contratos: Lista de los contratos que el usuario seleccionó desde el widget Multiselect
            lst_contratos = st.multiselect("Selecciona los contratos a incluir:", 
                                           options=df_actual['Contract'].unique(), 
                                           default=df_actual['Contract'].unique())
            df_dinamico = df_actual[df_actual['Contract'].isin(lst_contratos)]
            
        st.write(f"Filas resultantes en la tabla tras el filtro: **{df_dinamico.shape[0]}**")
        st.dataframe(df_dinamico.head(10))

    with tab10:
        st.header("Ítem 10: Hallazgos clave (Insights del EDA)")
        st.markdown("### Resumen Global de la Fuga (Churn)")
        
        # fig_pastel, ax_pastel: Componentes para generar el gráfico de proporciones global circular
        fig_pastel, ax_pastel = plt.subplots(figsize=(4, 4))
        df_actual['Churn'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax_pastel, colors=['#a8d5e2', '#f9a8a8'])
        ax_pastel.set_ylabel('')
        ax_pastel.set_title('Tasa Histórica de Churn (Yes/No)')
        st.pyplot(fig_pastel)
        
        st.markdown("""
        **Hallazgos Visuales Principales:**
        1. **Tenure y Lealtad:** A mayor permanencia del cliente en la empresa (tenure), el riesgo de churn disminuye drásticamente.
        2. **Contratos Flexibles:** El tipo de contrato 'Month-to-month' (mes a mes) lidera históricamente el conteo de bajas, denotando vulnerabilidad.
        3. **Facturación Elevada:** Los clientes que se fugan tienen medianas de cargos mensuales (`MonthlyCharges`) superiores a quienes permanecen en la empresa.
        """)

    with tab11:
        st.header("Diccionario de Datos")
        st.write("**Diccionario de Datos x Campo : **")
        st.write("custoimerID - Identificador único del cliente")
        st.write("gender - Género del cliente")
        st.write("SeniorCitizen - Si el Cliente es adultoMayor")
        st.write("Partner - Si el Cliente tiene pareja")
        st.write("Dependens - Si el Cliente tiene dependientes")
        st.write("ternure - Meses de permanencia")        
        st.write("PhoneService - Si el Cliente tiene Servicio Telefónico")        
        st.write("MultipleLines - Si tiene mpultiples lineas")        
        st.write("InternetService - Tipo de servicio de Internet")        
        st.write("OnlyneSecurity - Si posee seguridad en linea")        
        st.write("OnlineBackup - Si posee respaldo en línea")   
        st.write("DeviceProtection - Protección del dispositivo")   
        st.write("TechSupport - Soporte técnico")   
        st.write("StreamingTV - Servicio de TV")   
        st.write("StreamingMovies - Servicio de películas")   
        st.write("Contract - Tipo de contrato")   
        st.write("PaperlessBilling - Facturación electrónica")   
        st.write("PaymentMethod - Método de pago")   
        st.write("MonthlyCharges - Cargo mensual")   
        st.write("TotalCharges - Cargo total")   
        st.write("Churn - Si el cliente abandonó la empresa (“Yes”/“No”)")   
        st.markdown("" 
                    Glosario de Términos : 
                         - Media (Promedio): Suma de todos los valores dividida por el número total de datos. Se usa para describir el centro de un conjunto de datos con una distribución numérica normal y sin muchos valores atípicos.
                         - Mediana (Valor central): Es el valor intermedio cuando los datos están ordenados. Se utiliza como medida de tendencia central cuando los datos están sesgados o tienen valores atípicos, ya que no se ve afectada por ellos.
                         - Moda (Valor más común): Representa el valor que ocurre con mayor frecuencia en el conjunto de datos.
                         - Desviación Estándar: Mide qué tan dispersos están los valores en relación con la media.
                                   Baja: Los datos están cerca de la media (alta consistencia).
                                   Alta: Los datos están muy dispersos (alta variabilidad)
                    "")
        
# ==========================================
# MÓDULO 4: CONCLUSIONES
# ==========================================
def mostrar_conclusiones():
    st.title("Conclusiones y Recomendaciones de Negocio")
    st.markdown("""
    En base a los resultados del Análisis Exploratorio de Datos, establecemos los siguientes 5 puntos orientados a la toma de decisiones estratégicas de retención:
    
    1. **Migración de Contratos:** Existe una severa dependencia de la fuga en aquellos contratos *Month-to-Month*. Se deben incentivar estrategias de "up-selling", ofreciendo beneficios exclusivos (como descuentos iniciales) para que migren a contratos de un año.
    2. **Fidelización por Antigüedad:** El mayor volumen de abandonos ocurre durante el periodo de "tenure" más bajo (clientes nuevos). Se requiere instaurar programas de adopción, bienvenida y encuestas de satisfacción durante los primeros tres meses.
    3. **Servicios de Adhesión (Sticky Services):** Clientes que no poseen *OnlineSecurity* o *TechSupport* son mucho más propensos a la fuga. Se recomienda empaquetar estos servicios de seguridad a bajo costo (o gratis los primeros meses) para generar dependencia y aumentar los costos de cambio.
    4. **Métodos de Pago:** El método *Electronic Check* es significativamente riesgoso frente a medios automáticos. Establecer cargos o beneficios asociados a métodos de pago recurrentes automáticos (tarjetas de crédito domiciliadas) disminuirá las fugas por "olvido" o conveniencia.
    5. **Optimización de Precios (`MonthlyCharges`):** Clientes con altas tarifas mensuales presentan mayor propensión al Churn. Se recomienda segmentar proactivamente la base de clientes de tarifas altas y realizar ofertas preventivas o análisis competitivos de mercado antes de que decidan cancelar.
    """)

# Punto de entrada de la aplicación
if __name__ == "__main__":
    main()
