from pickle import load
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Cargar modelo
try:
    with open("nn_6_auto_cosine.model", 'rb') as model_file:
        model = load(model_file)
except FileNotFoundError:
    st.error("El archivo del modelo no se encontró.")
    st.stop()
except EOFError:
    st.error("Error al cargar el modelo: El archivo está vacío o dañado.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Cargar datos
df = pd.read_excel("../datos_merged_1986_2023.xlsx")

# Crear columnas adicionales para usar en el modelo
df["year_s"] = df["year"].astype(str)
df["duration_ms_s"] = df["duration_ms"].astype(str)
df["popularity_s"] = df["popularity"].astype(str)

df["tags"] = df["popularity_s"] + " " + df["duration_ms_s"] + " " + df["year_s"] + " " + df["artist_genres"]
df['tags'] = df['tags'].apply(lambda x: str(x).replace(";", " "))

# Crear matriz TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["tags"])

# Configuración de la aplicación con Streamlit
st.title("Recomendador de Canciones")

# Interfaz de usuario
cancion_input = st.text_input("Ingresa el nombre de la canción:", "Ej: Santeria")

# Lógica del recomendador
def lista_canciones(cancion):
    try:
        indice_cancion = df[df["track_name"] == cancion].index[0]
    except IndexError:
        st.error("Esa cancion es muy fantasma, intenta Usar Mayuscula En La Primera Letra.")
        st.stop()
    distancia, indices = model.kneighbors(tfidf_matrix[indice_cancion])
    canciones_similares = [(df["track_name"][i], distancia[0][j]) for j, i in enumerate(indices[0])]
    return canciones_similares[1:]

def str_canciones_recomendadas(cancion_input):
    recomendaciones = lista_canciones(cancion_input)
    resultado = "Recomendaciones para " + cancion_input + "<br />"
    for i, (cancion, distancia) in enumerate(recomendaciones, start=1):
        resultado += f"{i}. {cancion}<br />"
    return resultado

if st.button("Obtener Recomendaciones"):
    recomendaciones = str_canciones_recomendadas(cancion_input)
    st.markdown(recomendaciones, unsafe_allow_html=True)