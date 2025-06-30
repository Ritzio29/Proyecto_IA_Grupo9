from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from sklearn.metrics import accuracy_score

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

# Optimización de hiperparámetros con GridSearchCV
# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)
rf = grid_search_rf.best_estimator_
print(f"   ✅ Random Forest entrenado con mejores hiperparámetros: {grid_search_rf.best_params_}")

# SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_search_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=3, n_jobs=-1, verbose=1)
grid_search_svm.fit(X_train, y_train)
svm = grid_search_svm.best_estimator_
print(f"   ✅ SVM entrenado con mejores hiperparámetros: {grid_search_svm.best_params_}")

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
grid_search_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), param_grid_xgb, cv=3, n_jobs=-1, verbose=1)
grid_search_xgb.fit(X_train, y_train)
xgb = grid_search_xgb.best_estimator_
print(f"   ✅ XGBoost entrenado con mejores hiperparámetros: {grid_search_xgb.best_params_}")

# Naive Bayes (no tiene hiperparámetros para optimizar con GridSearchCV en este caso)
nb = GaussianNB()
nb.fit(X_train, y_train)
print("   ✅ Naive Bayes entrenado")

# Ensemble con VotingClassifier (se mantiene soft voting por defecto, la estrategia dinámica se implementa en predict)
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

        # Predicciones individuales y confianzas
        print("🤖 Calculando predicciones individuales y confianzas...")
        
        # Función auxiliar para obtener la confianza de un modelo
        def get_confidence(model, input_data):
            try:
                # Asegurarse de que el modelo puede predecir probabilidades
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(input_data)[0]
                    return float(probs.max())
                else:
                    # Si el modelo no tiene predict_proba (ej. algunos SVM sin probability=True)
                    # o si hay un problema, devolver 0 o un valor por defecto
                    return 0.0
            except Exception as e:
                print(f"⚠️ Advertencia: Error al obtener confianza para el modelo {type(model).__name__}: {e}")
                return 0.0 # Devolver 0.0 en caso de error

        rf_confidence = get_confidence(rf, responses_np)
        svm_confidence = get_confidence(svm, responses_np)
        xgb_confidence = get_confidence(xgb, responses_np)
        nb_confidence = get_confidence(nb, responses_np)

        preds = {
            'randomForest': rf_confidence,
            'svm': svm_confidence,
            'xgboost': xgb_confidence,
            'naiveBayes': nb_confidence,
        }
        print(f"📈 Confianzas individuales: {preds}")

        # Estrategia de votación inteligente y dinámica
        # Se puede definir un umbral o una lógica más compleja
        # Por simplicidad, si la confianza promedio de los modelos individuales es alta, usamos soft voting
        # Si es baja, podríamos considerar hard voting o un modelo específico
        
        # Calcular la confianza promedio de los modelos individuales, filtrando los 0.0 si se desea
        valid_confidences = [c for c in [rf_confidence, svm_confidence, xgb_confidence, nb_confidence] if c > 0]
        if valid_confidences:
            avg_individual_confidence = np.mean(valid_confidences)
        else:
            avg_individual_confidence = 0.0 # Si no hay confianzas válidas, la confianza promedio es 0

        print(f"📊 Confianza promedio individual: {avg_individual_confidence}")

        # Umbral para decidir entre soft y hard voting
        confidence_threshold = 0.75 # Este umbral puede ser ajustado

        if avg_individual_confidence >= confidence_threshold:
            print("🗳️ Usando Soft Voting debido a alta confianza individual promedio.")
            ensemble_probs = voting.predict_proba(responses_np)[0]
            final_index = ensemble_probs.argmax()
            final_label = label_encoder.inverse_transform([final_index])[0]
            avg_confidence = float(ensemble_probs[final_index])
            voting_strategy_used = "soft"
        else:
            print("🗳️ Usando Hard Voting debido a baja confianza individual promedio.")
            # Para hard voting, necesitamos las predicciones de clase de cada modelo
            rf_pred = rf.predict(responses_np)[0]
            svm_pred = svm.predict(responses_np)[0]
            xgb_pred = xgb.predict(responses_np)[0]
            nb_pred = nb.predict(responses_np)[0]

            # Votación de las clases predichas
            from collections import Counter
            hard_votes = [rf_pred, svm_pred, xgb_pred, nb_pred]
            most_common_vote = Counter(hard_votes).most_common(1)[0][0]
            final_label = label_encoder.inverse_transform([most_common_vote])[0]
            
            # Para la confianza en hard voting, podemos usar la proporción de votos
            avg_confidence = Counter(hard_votes)[most_common_vote] / len(hard_votes)
            voting_strategy_used = "hard"

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
                    "confidence": round(float(preds['randomForest']), 4)
                },
                "svm": {
                    "confidence": round(float(preds['svm']), 4)
                },
                "xgboost": {
                    "confidence": round(float(preds['xgboost']), 4)
                },
                "naiveBayes": {
                    "confidence": round(float(preds['naiveBayes']), 4)
                },
            },
            "votingStrategy": voting_strategy_used
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