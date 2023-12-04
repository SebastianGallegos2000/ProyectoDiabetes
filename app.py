import tkinter as tk
from tkinter import ttk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

class DiabetesPredictorApp:
    def __init__(self, root, df):
        self.root = root
        self.root.title("Diabetes Predictor")

        # Cargar y preprocesar los datos
        self.load_and_preprocess_data(df)

        # Crear la interfaz gráfica
        self.create_gui()

    def load_and_preprocess_data(self, df):
        # Eliminar variables que no entregan información importante
        X = df.drop(columns=['Education', 'Income', 'Diabetes_binary','AnyHealthcare','NoDocbcCost'])

        # Separar características y etiquetas
        Y = df['Diabetes_binary']

        # Dividir datos en conjunto de entrenamiento y prueba
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Aplicar SMOTE para abordar el desbalance de clases
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

        # Agregar nombres de columnas a X_train_resampled
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)

        # Crear y entrenar el modelo KNN
        self.model = KNeighborsClassifier()
        self.model.fit(X_train_resampled, Y_train_resampled)

    def create_gui(self):
        # Crear etiquetas y controles para las características
        self.labels = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                       'HvyAlcoholConsump', 'GenHlth',
                       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

        self.entry_vars = [tk.StringVar() for _ in range(len(self.labels))]

        for i, label in enumerate(self.labels):
            # Etiqueta a la izquierda
            ttk.Label(self.root, text=self.get_description(label), justify="left").grid(row=i, column=0, padx=10, pady=5)
            
            # Controles a la derecha
            # Controles específicos para algunas entradas
            if label == 'HighBP' or label == 'HighChol' or label == 'CholCheck' or label == 'Stroke' or label == 'HvyAlcoholConsump' or label == 'DiffWalk':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'BMI':
                ttk.Entry(self.root, textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'GenHlth':
                ttk.Combobox(self.root, values=['1', '2', '3', '4', '5'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'Sex':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'Smoker':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'HeartDiseaseorAttack':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'PhysActivity':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'Fruits':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'Veggies':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'MentHlth':
                ttk.Entry(self.root, textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'PhysHlth':
                ttk.Entry(self.root, textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            elif label == 'DiffWalk':
                ttk.Combobox(self.root, values=['0', '1'], textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')
            else:
                ttk.Entry(self.root, textvariable=self.entry_vars[i]).grid(row=i, column=1, padx=10, pady=5, sticky='e')

        # Botón de predicción
        self.predict_button = ttk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.labels), column=0, columnspan=3, pady=10)

    def predict(self):
        # Obtener las entradas del usuario y convertirlas a números
        features = [float(var.get()) if var.get().isdigit() else var.get() for var in self.entry_vars]

        # Realizar la predicción con el modelo
        prediction = self.model.predict([features])

        # Mostrar la predicción en una etiqueta
        result_label = ttk.Label(self.root, text=f"Prediction: {prediction[0]}")
        result_label.grid(row=len(self.labels) + 1, column=0, columnspan=3, pady=10)

    def get_description(self, label):
        # Retorna la descripción de la característica según el nombre de la columna
        descriptions = {
            'HighBP': 'Presión alta',
            'HighChol': 'Colesterol Alto',
            'CholCheck': 'Control de Colesterol en 5 años',
            'BMI':'Indice de masa corporal',
            'Smoker': 'Fumador',
            'Stroke': 'Derrame Cerebral',
            'HeartDiseaseorAttack':'Ataque al corazón',
            'PhysActivity':'Actividad física',
            'Fruits':'Consume frutas',
            'Veggies':'Consume verduras',
            'HvyAlcoholConsump': 'Consumo Excesivo de Alcohol',
            'GenHlth':'Como te sientes de salud general 1 excelente y 5 pésimo',
            'MentHlth': 'Días de Mala Salud Mental en los últimos 30 días, del 1 al 30',
            'PhysHlth': 'Días de Mala Salud Física en los últimos 30 días, del 1 al 30',
            'DiffWalk': 'Dificultades para Caminar o Subir Escaleras',
            'Sex':'Sexo',
            'Age':'Edad'
        }
        return descriptions.get(label, label)

# Cargar los datos desde un archivo local
# Asegúrate de proporcionar la ruta correcta al archivo CSV
csv_path = "diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(csv_path)

# Crear la instancia de la aplicación y ejecutar el bucle principal
root = tk.Tk()
app = DiabetesPredictorApp(root, df)
root.mainloop()
