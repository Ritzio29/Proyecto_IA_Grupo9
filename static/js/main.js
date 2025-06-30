function iniciarSesion() {
    document.getElementById("student-section").classList.add("hidden");
    document.getElementById("admin-section").classList.add("hidden");

    const user = document.getElementById("loginUser").value;
    const pass = document.getElementById("loginPass").value;

    const loginSection = document.getElementById("login-section");
    const studentSection = document.getElementById("student-section");
    const adminSection = document.getElementById("admin-section");


    if (user === "admin" && pass === "1234") {
        loginSection.classList.add("hidden");
        adminSection.classList.remove("hidden");
        initializeCharts(); // reinicializar gráficos al mostrar admin

    } else if (user === "estudiante" && pass === "abcd") {
        loginSection.classList.add("hidden");
        studentSection.classList.remove("hidden");
    } else {
        alert("Usuario o contraseña incorrectos.");
    }

}

// Variables globales
let currentResults = null;

// Navegación entre secciones
function showSection(section) {
    const studentSection = document.getElementById('student-section');
    const adminSection = document.getElementById('admin-section');

    if (section === 'student') {
        studentSection.classList.remove('hidden');
        adminSection.classList.add('hidden');
    } else {
        studentSection.classList.add('hidden');
        adminSection.classList.remove('hidden');
        initializeCharts();
    }
}

// Simulación de agentes del sistema multiagente
class MultiAgentSystem {
    constructor() {
        this.agents = {
            extractor: new ExtractorAgent(),
            preprocessor: new PreprocessorAgent(),
            optimizer: new OptimizerAgent(),
            ensemble: new EnsembleAgent(),
            visualizer: new VisualizerAgent()
        };
    }

    async processResponse(responses) {
        console.log("🤖 Iniciando Sistema Multiagente PHQ-9...");

        // Agente Extractor
        const extractedData = await this.agents.extractor.extract(responses);

        // Agente Preprocesador
        const processedData = await this.agents.preprocessor.process(extractedData);

        // Agente Optimizador
        const models = await this.agents.optimizer.optimizeModels(processedData);

        // Agente Ensemble
        const prediction = await this.agents.ensemble.predict(processedData, models);

        // Agente Visualizador
        const visualization = await this.agents.visualizer.visualize(prediction);

        return {
            ...prediction,
            visualization
        };
    }
}

class ExtractorAgent {
    async extract(responses) {
        console.log("📥 Agente Extractor: Procesando respuestas PHQ-9...");
        return {
            responses: responses,
            timestamp: new Date(),
            format: 'structured_dataframe'
        };
    }
}

class PreprocessorAgent {
    async process(data) {
        console.log("🧹 Agente Preprocesador: Limpieza y normalización...");
        return {
            ...data,
            normalized: true,
            missing_values_handled: true,
            encoded: true
        };
    }
}

class OptimizerAgent {
    async optimizeModels(data) {
        console.log("⚙️ Agente Optimizador: Ajustando hiperparámetros...");
        return {
            randomForest: { accuracy: 0.948, confidence: 0.92 },
            svm: { accuracy: 0.931, confidence: 0.88 },
            xgboost: { accuracy: 0.952, confidence: 0.94 },
            naiveBayes: { accuracy: 0.897, confidence: 0.82 }
        };
    }
}

class EnsembleAgent {
    async predict(data, models) {
        console.log("🔗 Agente Ensemble: Combinando predicciones...");

        const totalScore = data.responses.reduce((sum, score) => sum + parseInt(score), 0);
        let classification, explanation = [];

        // Clasificación según PHQ-9
        if (totalScore <= 4) {
            classification = "Mínima";
        } else if (totalScore <= 9) {
            classification = "Leve";
        } else if (totalScore <= 14) {
            classification = "Moderada";
        } else if (totalScore <= 19) {
            classification = "Moderadamente Severa";
        } else {
            classification = "Severa";
        }

        // Explicabilidad basada en respuestas altas
        data.responses.forEach((score, index) => {
            if (parseInt(score) >= 2) {
                explanation.push(`Pregunta ${index + 1}: Puntuación alta (${score})`);
            }
        });

        const avgConfidence = Object.values(models).reduce((sum, model) => sum + model.confidence, 0) / 4;

        return {
            score: totalScore,
            classification: classification,
            confidence: avgConfidence,
            explanation: explanation,
            models: models,
            votingStrategy: avgConfidence > 0.85 ? 'soft' : 'hard'
        };
    }
}

class VisualizerAgent {
    async visualize(prediction) {
        console.log("📊 Agente Visualizador: Generando visualizaciones...");
        return {
            charts: ['confidence_chart', 'explanation_chart'],
            reports: ['pdf_ready', 'excel_ready']
        };
    }
}

// Inicializar sistema
const multiAgentSystem = new MultiAgentSystem();

// Manejo del formulario
document.getElementById('phq9-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const responses = [];

    for (let i = 1; i <= 9; i++) {
        const value = formData.get(`q${i}`);
        if (value === null) {
            alert(`Por favor responde la pregunta ${i}`);
            return;
        }
        responses.push(parseInt(value));
    }

    // Mostrar loading
    const button = e.target.querySelector('button[type="submit"]');
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Procesando con Sistema Multiagente...';
    button.disabled = true;

    try {
        // ✅ MEJORAR el manejo de la petición
        const response = await fetch('https://proyecto-ia-grupo9-p2se.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ responses })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        currentResults = result;
        displayResults(result);

    } catch (error) {
        console.error('Error en el sistema multiagente:', error);
        alert('Error al procesar las respuestas. Verifica que el servidor esté ejecutándose en http://127.0.0.1:5000');
    } finally {
        button.innerHTML = originalText;
        button.disabled = false;
    }
});

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const resultCard = document.getElementById('result-card');

    // Aplicar clase CSS según clasificación
    resultCard.className = `result-card ${result.classification.toLowerCase().replace(/\s+/g, '-')}`;

    document.getElementById('classification').textContent = result.classification;
    document.getElementById('score').textContent = `${result.score}/27 puntos`;

    // Barra de confianza
    const confidencePercent = Math.round(result.confidence * 100);
    document.getElementById('confidence-bar').style.width = `${confidencePercent}%`;
    document.getElementById('confidence-text').textContent = `${confidencePercent}% de confianza (${result.votingStrategy} voting)`;

    // Explicación
    const explanationDiv = document.getElementById('explanation');
    explanationDiv.innerHTML = `
                <p><strong>Estrategia de Voting:</strong> ${result.votingStrategy} voting seleccionado automáticamente</p>
                <p><strong>Modelos utilizados:</strong> Random Forest (${Math.round(result.models.randomForest.confidence * 100)}%), 
                SVM (${Math.round(result.models.svm.confidence * 100)}%), 
                XGBoost (${Math.round(result.models.xgboost.confidence * 100)}%), 
                Naive Bayes (${Math.round(result.models.naiveBayes.confidence * 100)}%)</p>
                ${result.explanation.length > 0 ?
            `<p><strong>Factores principales:</strong></p><ul class="list-disc list-inside">${result.explanation.map(exp => `<li>${exp}</li>`).join('')}</ul>`
            : '<p>No se identificaron factores de riesgo significativos.</p>'}
            `;

    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Funciones de exportación
function exportPDF() {
    if (!currentResults) return;

    const printContent = document.getElementById('results').innerHTML;
    const originalContent = document.body.innerHTML;

    document.body.innerHTML = `
                <div style="font-family: Arial, sans-serif; padding: 20px;">
                    <h1>Reporte PHQ-9 - Sistema Multiagente</h1>
                    <p>Fecha: ${new Date().toLocaleDateString()}</p>
                    ${printContent}
                </div>
            `;

    window.print();
    document.body.innerHTML = originalContent;
    location.reload();
}

function exportExcel() {
    if (!currentResults) return;

    const data = [
        ['Fecha', new Date().toLocaleDateString()],
        ['Puntuación PHQ-9', currentResults.score],
        ['Clasificación', currentResults.classification],
        ['Confianza', Math.round(currentResults.confidence * 100) + '%'],
        ['Estrategia Voting', currentResults.votingStrategy], // Corregido: era votingStrateg
        ['Random Forest', Math.round(currentResults.models.randomForest.confidence * 100) + '%'],
        ['SVM', Math.round(currentResults.models.svm.confidence * 100) + '%'],
        ['XGBoost', Math.round(currentResults.models.xgboost.confidence * 100) + '%'],
        ['Naive Bayes', Math.round(currentResults.models.naiveBayes.confidence * 100) + '%']
    ];

    let csvContent = "data:text/csv;charset=utf-8,";
    data.forEach(row => {
        csvContent += row.join(",") + "\r\n";
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `reporte_phq9_${new Date().getTime()}.csv`);
    link.click();
}

// Inicializar gráficos del panel administrativo
function initializeCharts() {
    // Gráfico de distribución de depresión
    const ctx1 = document.getElementById('depressionChart');
    if (ctx1) {
        new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Mínima', 'Leve', 'Moderada', 'Mod. Severa', 'Severa'],
                datasets: [{
                    data: [89, 76, 48, 23, 11],
                    backgroundColor: [
                        '#10b981',
                        '#f59e0b',
                        '#f97316',
                        '#ef4444',
                        '#dc2626'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Gráfico de rendimiento de modelos
    const ctx2 = document.getElementById('modelsChart');
    if (ctx2) {
        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ['Random Forest', 'SVM', 'XGBoost', 'Naive Bayes'],
                datasets: [{
                    label: 'Precisión (%)',
                    data: [94.8, 93.1, 95.2, 89.7],
                    backgroundColor: [
                        '#3b82f6',
                        '#8b5cf6',
                        '#10b981',
                        '#f59e0b'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
}

function volverAlLogin() {
    document.getElementById("login-section").classList.remove("hidden");
    document.getElementById("student-section").classList.add("hidden");
    document.getElementById("admin-section").classList.add("hidden");
    document.getElementById("results").classList.add("hidden");

    // Limpiar campos
    document.getElementById("loginUser").value = "";
    document.getElementById("loginPass").value = "";
    document.getElementById("phq9-form").reset();
}