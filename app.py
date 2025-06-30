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
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ✅ CONFIGURACIÓN CORS SIMPLIFICADA - Solo una configuración
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"])

# Cargar el nuevo dataset numérico
try:
    df = pd.read_csv("PHQ9_Dataset.csv", encoding='utf-8')
    print(f"✅ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
    print(f"📋 Columnas disponibles: {list(df.columns)}")
    
    # 🔍 DEBUGGING: Mostrar primeras filas
    print(f"📊 Primeras 5 filas del dataset:")
    print(df.head())
    
    # 🔍 DEBUGGING: Verificar tipos de datos
    print(f"📊 Tipos de datos:")
    print(df.dtypes)
    
    # 🔍 DEBUGGING: Verificar valores nulos
    print(f"📊 Valores nulos por columna:")
    print(df.isnull().sum())
    
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
    
    # 🔍 DEBUGGING: Verificar valores en X
    print(f"📊 Estadísticas de X:")
    print(X.describe())
    
    # 🔍 DEBUGGING: Verificar si hay valores infinitos o NaN en X
    print(f"📊 Valores NaN en X: {X.isnull().sum().sum()}")
    print(f"📊 Valores infinitos en X: {np.isinf(X.values).sum()}")
    
except KeyError as e:
    print(f"❌ ERROR: Columna no encontrada: {e}")
    print("📋 Columnas disponibles en el dataset:")
    for col in df.columns:
        print(f"   - {col}")
    exit(1)

# Crear etiquetas de severidad a partir del PHQ-9 Score
def score_to_severity(score):
    if pd.isna(score):  # 🔍 Manejo de valores NaN
        return "Desconocido"
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
    # 🔍 DEBUGGING: Verificar la columna PHQ-9 Score
    print(f"📊 Valores únicos en PHQ-9 Score: {sorted(df['PHQ-9 Score'].unique())}")
    print(f"📊 Valores NaN en PHQ-9 Score: {df['PHQ-9 Score'].isnull().sum()}")
    
    df["Severity Level"] = df["PHQ-9 Score"].apply(score_to_severity)
    y = df["Severity Level"]
    print(f"✅ Etiquetas creadas: {y.value_counts().to_dict()}")
    
    # 🔍 DEBUGGING: Verificar si hay etiquetas "Desconocido"
    if "Desconocido" in y.values:
        print("⚠️ ADVERTENCIA: Se encontraron valores NaN en PHQ-9 Score")
        
except KeyError:
    print("❌ ERROR: No se encontró la columna 'PHQ-9 Score'")
    print("📋 Columnas disponibles:")
    for col in df.columns:
        print(f"   - {col}")
    exit(1)

# 🔍 DEBUGGING: Limpiar datos antes del entrenamiento
print("🧹 Limpiando datos...")

# Eliminar filas con valores NaN en X o y
mask = ~(X.isnull().any(axis=1) | y.isnull())
X_clean = X[mask]
y_clean = y[mask]

print(f"📊 Datos después de limpieza: {len(X_clean)} filas (eliminadas: {len(X) - len(X_clean)})")

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_clean)

print(f"📊 Clases codificadas: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Dividir en entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"✅ División completada: Train={len(X_train)}, Test={len(X_test)}")

# Entrenar modelos con parámetros más conservadores
print("🤖 Entrenando modelos...")

# Random Forest con parámetros más simples
print("   🌳 Entrenando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"   ✅ Random Forest entrenado - Accuracy: {rf_accuracy:.4f}")

# SVM con parámetros más simples
print("   🎯 Entrenando SVM...")
svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm.predict(X_test))
print(f"   ✅ SVM entrenado - Accuracy: {svm_accuracy:.4f}")

# XGBoost con parámetros más simples
print("   🚀 Entrenando XGBoost...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_accuracy = accuracy_score(y_test, xgb.predict(X_test))
print(f"   ✅ XGBoost entrenado - Accuracy: {xgb_accuracy:.4f}")

# Naive Bayes
print("   📊 Entrenando Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_accuracy = accuracy_score(y_test, nb.predict(X_test))
print(f"   ✅ Naive Bayes entrenado - Accuracy: {nb_accuracy:.4f}")

# Ensemble con VotingClassifier
print("   🗳️ Creando Ensemble...")
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
voting_accuracy = accuracy_score(y_test, voting.predict(X_test))
print(f"   ✅ Ensemble entrenado - Accuracy: {voting_accuracy:.4f}")

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
        
        # 🔍 DEBUGGING: Verificar el array de entrada
        print(f"📊 Shape del array: {responses_np.shape}")
        print(f"📊 Valores del array: {responses_np}")
        print(f"📊 Hay NaN en el array: {np.isnan(responses_np).any()}")
        print(f"📊 Hay infinitos en el array: {np.isinf(responses_np).any()}")
        
        # Calcular el score total
        total_score = sum(responses)
        print(f"📊 Score total: {total_score}")

        # Predicciones individuales y confianzas
        print("🤖 Calculando predicciones individuales y confianzas...")
        
        try:
            # Obtener probabilidades de cada clasificador
            print("   🌳 Predicción Random Forest...")
            rf_probs = rf.predict_proba(responses_np)[0]
            print(f"   📊 RF probabilidades: {rf_probs}")
            
            print("   🎯 Predicción SVM...")
            svm_probs = svm.predict_proba(responses_np)[0]
            print(f"   📊 SVM probabilidades: {svm_probs}")
            
            print("   🚀 Predicción XGBoost...")
            xgb_probs = xgb.predict_proba(responses_np)[0]
            print(f"   📊 XGB probabilidades: {xgb_probs}")
            
            print("   📊 Predicción Naive Bayes...")
            nb_probs = nb.predict_proba(responses_np)[0]
            print(f"   📊 NB probabilidades: {nb_probs}")

            # 🔍 DEBUGGING: Verificar si hay NaN en las probabilidades
            if np.isnan(rf_probs).any():
                print("⚠️ Random Forest devolvió NaN")
            if np.isnan(svm_probs).any():
                print("⚠️ SVM devolvió NaN")
            if np.isnan(xgb_probs).any():
                print("⚠️ XGBoost devolvió NaN")
            if np.isnan(nb_probs).any():
                print("⚠️ Naive Bayes devolvió NaN")

            # Confianzas individuales (máxima probabilidad predicha)
            rf_confidence = float(rf_probs.max()) if not np.isnan(rf_probs).any() else 0.0
            svm_confidence = float(svm_probs.max()) if not np.isnan(svm_probs).any() else 0.0
            xgb_confidence = float(xgb_probs.max()) if not np.isnan(xgb_probs).any() else 0.0
            nb_confidence = float(nb_probs.max()) if not np.isnan(nb_probs).any() else 0.0

            preds = {
                'randomForest': rf_confidence,
                'svm': svm_confidence,
                'xgboost': xgb_confidence,
                'naiveBayes': nb_confidence,
            }
            print(f"📈 Confianzas individuales: {preds}")

            # Estrategia de votación inteligente y dinámica
            valid_confidences = [conf for conf in [rf_confidence, svm_confidence, xgb_confidence, nb_confidence] if not np.isnan(conf) and conf > 0]
            
            if len(valid_confidences) == 0:
                print("❌ Todas las confianzas son NaN o 0")
                return jsonify({"error": "Error en la predicción de todos los modelos"}), 500
            
            avg_individual_confidence = np.mean(valid_confidences)
            print(f"📊 Confianza promedio individual: {avg_individual_confidence}")

            # Umbral para decidir entre soft y hard voting
            confidence_threshold = 0.75

            if avg_individual_confidence >= confidence_threshold:
                print("🗳️ Usando Soft Voting debido a alta confianza individual promedio.")
                ensemble_probs = voting.predict_proba(responses_np)[0]
                if np.isnan(ensemble_probs).any():
                    print("⚠️ Ensemble devolvió NaN, usando modelo individual mejor")
                    best_model_idx = np.argmax([rf_confidence, svm_confidence, xgb_confidence, nb_confidence])
                    models = [rf, svm, xgb, nb]
                    best_probs = models[best_model_idx].predict_proba(responses_np)[0]
                    final_index = best_probs.argmax()
                    avg_confidence = float(best_probs[final_index])
                else:
                    final_index = ensemble_probs.argmax()
                    avg_confidence = float(ensemble_probs[final_index])
                voting_strategy_used = "soft"
            else:
                print("🗳️ Usando Hard Voting debido a baja confianza individual promedio.")
                rf_pred = rf.predict(responses_np)[0]
                svm_pred = svm.predict(responses_np)[0]
                xgb_pred = xgb.predict(responses_np)[0]
                nb_pred = nb.predict(responses_np)[0]

                from collections import Counter
                hard_votes = [rf_pred, svm_pred, xgb_pred, nb_pred]
                most_common_vote = Counter(hard_votes).most_common(1)[0][0]
                final_index = most_common_vote
                avg_confidence = Counter(hard_votes)[most_common_vote] / len(hard_votes)
                voting_strategy_used = "hard"

            final_label = label_encoder.inverse_transform([final_index])[0]
            print(f"🎯 Predicción final: {final_label} (confianza: {avg_confidence})")

        except Exception as model_error:
            print(f"💥 ERROR en predicción de modelos: {str(model_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error en predicción: {str(model_error)}"}), 500

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