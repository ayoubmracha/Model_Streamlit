import streamlit as st
import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
st.image("ehtp.png", caption=" ", use_column_width=True)
with open("modeliris6.pkl", "rb") as file:
    model = pickle.load(file)
st.title("MSDE6 : ML Course")
st.header("IRIS Flower predection App")
st.markdown("This app predicts the Iris flower type")

input_choice = st.selectbox('How would you like to use the prediction model ?', ['','Input parametrs directly', 'Load a file of data'])

# Afficher les sliders si l'utilisateur sélectionne "input parameters directly"
st.sidebar.image("flower.jpg", caption=" ", use_column_width=True)
if input_choice == "Input parametrs directly":

    st.sidebar.markdown("### User Input Parameters:")

    # Sliders dans la sidebar pour saisir les valeurs des caractéristiques de la fleur
    sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 6.0)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 5.0, 3.0)
    petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal width", 0.1, 3.0, 1.0)

    # Afficher les paramètres sous forme de tableau
    input_data = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })
    
    st.write("### User Input Parameters:")
    st.table(input_data)

    # Prédiction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Afficher la classe prédite et les probabilités
    st.write("### Prediction")
    st.write(f"Predicted Class: {prediction[0]}")

    st.write("### Prediction Probability")
    st.write(prediction_proba)

# Si l'utilisateur choisit de charger un fichier, on peut ajouter le code pour charger et afficher un fichier.
elif input_choice == "Load a file of data":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded File Data:")
        st.write(data)
        predictions = model.predict(data)
        prediction_proba = model.predict_proba(data)

        # Afficher les prédictions
        st.write("### Predictions")
        data["Prediction"] = predictions
        st.write(data)

        # Afficher les probabilités
        st.write("### Prediction Probability")
        st.write(prediction_proba)