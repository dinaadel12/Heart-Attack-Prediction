import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import joblib


# Load the model
model = joblib.load("ensembel_rf_model.pkl")

# Load the dataset
df = pd.read_csv('heart.csv')

# Split features and target variable
X = df.drop(columns='output')
y = df['output']

# Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling (optional but useful for logistic regression)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train logistic regression model
# log_model = LogisticRegression(max_iter=1000)
# log_model.fit(X_train, y_train)

# Dash App
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Heart Attack Prediction Dashboard", style={'text-align': 'center'}),

    # Correlation Matrix Heatmap
    dcc.Graph(id='correlation-matrix'),

    # Feature Importance Section
    dcc.Graph(id='feature-importance'),

    # User Input Section
    html.H2("Input Patient Data"),
    html.Div([
        html.Label('Age'),
        dcc.Input(id='input-age', type='number', value=50, step=1),

        html.Label('Sex (1=Male, 0=Female)'),
        dcc.Input(id='input-sex', type='number', value=1, step=1),

        html.Label('Chest Pain Type (0-3)'),
        dcc.Input(id='input-cp', type='number', value=2, step=1),

        html.Label('Resting Blood Pressure (trtbps)'),
        dcc.Input(id='input-trtbps', type='number', value=120, step=1),

        html.Label('Cholesterol Level (chol)'),
        dcc.Input(id='input-chol', type='number', value=250, step=1),

        html.Label('Fasting Blood Sugar (1=True, 0=False)'),
        dcc.Input(id='input-fbs', type='number', value=0, step=1),

        html.Label('Resting ECG (restecg, 0-2)'),
        dcc.Input(id='input-restecg', type='number', value=1, step=1),

        html.Label('Maximum Heart Rate (thalachh)'),
        dcc.Input(id='input-thalachh', type='number', value=150, step=1),

        html.Label('Exercise Induced Angina (exng, 1=True, 0=False)'),
        dcc.Input(id='input-exng', type='number', value=0, step=1),

        html.Label('Oldpeak'),
        dcc.Input(id='input-oldpeak', type='number', value=1.0, step=0.1),

        html.Label('Slope of Peak Exercise ST (slp, 0-2)'),
        dcc.Input(id='input-slp', type='number', value=1, step=1),

        html.Label('Number of Major Vessels (caa, 0-3)'),
        dcc.Input(id='input-caa', type='number', value=0, step=1),

        html.Label('Thalassemia (thall, 1-3)'),
        dcc.Input(id='input-thall', type='number', value=2, step=1)
    ], style={'columnCount': 2}),

    html.Button('Predict', id='predict-button'),
    html.H2("Prediction Result"),
    html.Div(id='prediction-output')
])

# Callbacks

# Correlation matrix visualization
@app.callback(
    Output('correlation-matrix', 'figure'),
    Input('correlation-matrix', 'id')
)
def update_corr_matrix(_):
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', title="Correlation Matrix")
    return fig

# Feature importance visualization
@app.callback(
    Output('feature-importance', 'figure'),
    Input('feature-importance', 'id')
)
def update_feature_importance(_):
    # model = LogisticRegression(max_iter=1000)
    #model.fit(X_train, y_train)

    #importance = np.abs(model.coef_[0])
    # features = df.drop(columns='output').columns
    features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'sex_0',
       'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0',
       'restecg_1', 'restecg_2', 'exng_0', 'exng_1', 'caa_0', 'caa_1', 'caa_2',
       'caa_3', 'caa_4', 'thall_0', 'thall_1', 'thall_2', 'thall_3', 'slp_0',
       'slp_1', 'slp_2']
    
    # Calculate permutation importance
    #results = permutation_importance(model, X_test, y_test, scoring='accuracy', n_repeats=30, random_state=0)

    # Extract feature importances
    #importances = results.importances_mean
    fig = px.bar(x=features, y=model.feature_importances_, title="Feature Importance")
    return fig

# Prediction based on user input
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        Input('input-age', 'value'),
        Input('input-sex', 'value'),
        Input('input-cp', 'value'),
        Input('input-trtbps', 'value'),
        Input('input-chol', 'value'),
        Input('input-fbs', 'value'),
        Input('input-restecg', 'value'),
        Input('input-thalachh', 'value'),
        Input('input-exng', 'value'),
        Input('input-oldpeak', 'value'),
        Input('input-slp', 'value'),
        Input('input-caa', 'value'),
        Input('input-thall', 'value')
    ]
)
def predict_heart_attack(n_clicks, age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    if n_clicks is not None:
        # Collect user input into a DataFrame
        #input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])

        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trtbps': [trtbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalachh': [thalachh],
            'exng': [exng],
            'oldpeak': [oldpeak],
            'slp': [slp],
            'caa': [caa],
            'thall': [thall]

        })


        #instantiating the scaler
        scaler = RobustScaler()
        neumerical_data = ['age', 'trtbps', 'chol',  'thalachh', 'oldpeak'  ]
        caterogrical_data = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'caa', 'thall','slp' ]

        input_data = pd.get_dummies(input_data, columns = caterogrical_data)

        # scaling the continuous featuree
        input_data[neumerical_data] = scaler.fit_transform(input_data[neumerical_data])

        columns = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'sex_0',
       'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0',
       'restecg_1', 'restecg_2', 'exng_0', 'exng_1', 'caa_0', 'caa_1', 'caa_2',
       'caa_3', 'caa_4', 'thall_0', 'thall_1', 'thall_2', 'thall_3', 'slp_0',
       'slp_1', 'slp_2']
        
        input_data = input_data.reindex(columns=columns, fill_value=0)

        # Reshape the input data to a 1-dimensional array
        input_data = input_data.values.reshape(1, -1)
        

        # Make prediction
        prediction = model.predict(input_data)
        return f"Predicted: {'Heart Attack Risk' if prediction[0] == 1 else 'No Heart Attack Risk'}"
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

