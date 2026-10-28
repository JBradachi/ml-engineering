from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

best_model = joblib.load("/home/bradas/Documentos/github/ml-engineering/atividades/atv7/xgb.joblib")

decoder = joblib.load("/home/bradas/Documentos/github/ml-engineering/atividades/atv7/encoder.joblib")

ordem_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/predict', methods=['POST'])
def predict():
    linha = request.get_json()
    df = pd.DataFrame([linha])
    df = df[ordem_features]

    previsao = best_model.predict(df)
    return {"previsao": decoder.inverse_transform(previsao)[0]}

if __name__ == '__main__':
    app.run(debug=True)
































