from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)

# ✅ CONFIGURACIÓN CORS SIMPLIFICADA - Solo una configuración
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"])


# Cargar el nuevo dataset numérico
try:
    df = pd.read_csv("PHQ9_Dataset.csv", encoding='utf-8')
    print(f"✅ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
    print(f"📋 Columnas disponibles: {list(df.columns)}")
except FileNotFoundError:
    print("❌ ERROR: No se encontró el archivo PHQ9_Dataset.csv")
    print("   Asegúrate de que el archivo esté en la misma carpeta que app.py")
    exit(1)
except Exception as e:
    print(f"❌ ERROR al cargar dataset: {e}")
    exit(1)

# Nombres de columnas de preguntas (asegúrate de que coincidan exactamente con tu CSV)
questions = [
    "Do you have little interest or pleasure in doing things?_Numeric",
    "Do you feel down, depressed, or hopeless?_Numeric",
    "Do you have trouble falling or staying asleep, or do you sleep too much?_Numeric",
    "Do you feel tired or have little energy?_Numeric",
    "Do you have poor appetite or tend to overeat?_Numeric",
    "Do you feel bad about yourself or that you are a failure or have let yourself or your family down?_Numeric",
    "Do you have trouble concentrating on things, such as reading, work, or watching television?_Numeric",
    "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?_Numeric",
    "Have you had thoughts of self-harm or felt that you would be better off dead?_Numeric"
]

# Usar las columnas numéricas como características (X)
try:
    X = df[questions]
    print(f"✅ Características extraídas: {X.shape}")
except KeyError as e:
    print(f"❌ ERROR: Columna no encontrada: {e}")
    print("📋 Columnas disponibles en el dataset:")
    for col in df.columns:
        print(f"   - {col}")
    exit(1)

# Crear etiquetas de severidad a partir del PHQ-9 Score
def score_to_severity(score):
    if score <= 4:
        return "Mínima"
    elif score <= 9:
        return "Leve"
    elif score <= 14:
        return "Moderada"
    elif score <= 19:
        return "Moderadamente Severa"
    else:
        return "Severa"

try:
    df["Severity Level"] = df["PHQ-9 Score"].apply(score_to_severity)
    y = df["Severity Level"]
    print(f"✅ Etiquetas creadas: {y.value_counts().to_dict()}")
except KeyError:
    print("❌ ERROR: No se encontró la columna 'PHQ-9 Score'")
    print("📋 Columnas disponibles:")
    for col in df.columns:
        print(f"   - {col}")
    exit(1)

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"✅ División completada: Train={len(X_train)}, Test={len(X_test)}")

# Entrenar modelos
print("🤖 Entrenando modelos...")
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
nb = GaussianNB()

rf.fit(X_train, y_train)
print("   ✅ Random Forest entrenado")
svm.fit(X_train, y_train)
print("   ✅ SVM entrenado")
xgb.fit(X_train, y_train)
print("   ✅ XGBoost entrenado")
nb.fit(X_train, y_train)
print("   ✅ Naive Bayes entrenado")

# Ensemble con VotingClassifier
voting = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('svm', svm),
        ('xgb', xgb),
        ('nb', nb)
    ],
    voting='soft'
)
voting.fit(X_train, y_train)
print("   ✅ Ensemble entrenado")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(f"📥 Recibiendo petición POST en /predict")
        data = request.get_json()
        print(f"📋 Datos recibidos: {data}")
        
        if not data or 'responses' not in data:
            print("❌ Error: Faltan datos - se requiere 'responses'")
            return jsonify({"error": "Faltan datos - se requiere 'responses'"}), 400

        responses = data['responses']
        print(f"📊 Respuestas: {responses}")
        
        # Validar que tenemos exactamente 9 respuestas
        if len(responses) != 9:
            print(f"❌ Error: Se esperaban 9 respuestas, se recibieron {len(responses)}")
            return jsonify({"error": "Se requieren exactamente 9 respuestas"}), 400
        
        # Convertir a integers y validar rango
        try:
            responses = [int(x) for x in responses]
            if not all(0 <= r <= 3 for r in responses):
                print(f"❌ Error: Respuestas fuera de rango: {responses}")
                return jsonify({"error": "Las respuestas deben estar entre 0 y 3"}), 400
        except ValueError as ve:
            print(f"❌ Error de conversión: {ve}")
            return jsonify({"error": "Las respuestas deben ser números enteros"}), 400

        responses_np = np.array([responses])
        print(f"🔢 Array numpy: {responses_np}")
        
        # Calcular el score total
        total_score = sum(responses)
        print(f"📊 Score total: {total_score}")

        # Predicciones individuales
        print("🤖 Calculando predicciones...")
        preds = {
            'randomForest': float(rf.predict_proba(responses_np)[0].max()),
            'svm': float(svm.predict_proba(responses_np)[0].max()),
            'xgboost': float(xgb.predict_proba(responses_np)[0].max()),
            'naiveBayes': float(nb.predict_proba(responses_np)[0].max()),
        }
        print(f"📈 Predicciones individuales: {preds}")

        # Predicción ensemble
        ensemble_probs = voting.predict_proba(responses_np)[0]
        final_index = ensemble_probs.argmax()
        final_label = label_encoder.inverse_transform([final_index])[0]
        avg_confidence = float(ensemble_probs[final_index])  # ✅ Convertir a float
        print(f"🎯 Predicción final: {final_label} (confianza: {avg_confidence})")

        # Explicación basada en respuestas altas
        explanation = []
        for i, val in enumerate(responses):
            if val >= 2:
                explanation.append(f"Pregunta {i+1}: Puntuación alta ({val})")

        result = {
            "score": total_score,
            "classification": final_label,
            "confidence": round(avg_confidence, 4),
            "explanation": explanation,
            "models": {
                "randomForest": {
                    "accuracy": round(float(rf.score(X_test, y_test)), 4),
                    "confidence": round(float(preds['randomForest']), 4)
                },
                "svm": {
                    "accuracy": round(float(svm.score(X_test, y_test)), 4),
                    "confidence": round(float(preds['svm']), 4)
                },
                "xgboost": {
                    "accuracy": round(float(xgb.score(X_test, y_test)), 4),
                    "confidence": round(float(preds['xgboost']), 4)
                },
                "naiveBayes": {
                    "accuracy": round(float(nb.score(X_test, y_test)), 4),
                    "confidence": round(float(preds['naiveBayes']), 4)
                },
            },
            "votingStrategy": "soft"
        }
        
        print(f"✅ Respuesta exitosa: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"💥 ERROR CRÍTICO en predict: {str(e)}")
        print(f"📋 Tipo de error: {type(e).__name__}")
        import traceback
        print(f"📍 Traceback completo:")
        traceback.print_exc()
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

# Ruta de prueba para verificar que el servidor funciona
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "Servidor funcionando correctamente"})
from flask import render_template

@app.route('/')
def home():
    return render_template("index.html")

# Ejecutar servidor
if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask...")
    print("📊 Modelos entrenados y listos")
    print("🌐 Servidor disponible en: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

    