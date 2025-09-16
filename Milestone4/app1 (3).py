import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import io
import base64
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

app = dash.Dash(__name__)
app.title = "Data Analysis App"

dfG = None
model = None
featOrder = []

app.layout = html.Div([
    html.H1("Data Analysis App", style={"textAlign": "center"}),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Upload File']),
            style={
                'width': '40%', 'height': '50px', 'lineHeight': '50px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px auto'
            }
        ),
        html.Div(id='output-data-upload', style={"margin": "10px"}),
    ], style={"textAlign": "left", "margin": "10px"}),

    html.Div([
        html.Label("Select Target:"),
        dcc.Dropdown(
            id='target-variable-dropdown',
            options=[],
            placeholder=" ",
            style={"width": "60%", "margin": "auto"}
        )
    ], style={"textAlign": "center", "margin": "20px 0"}),

    html.Div([
        html.Label(""),
        dcc.RadioItems(
            id='categorical-variable-radio',
            options=[],
            style={"margin": "auto"}
        )
    ], style={"textAlign": "center", "margin": "20px 0"}),

    html.Div([
        dcc.Graph(id='avg-value-chart', style={"display": "inline-block", "width": "48%"}),
        dcc.Graph(id='correlation-chart', style={"display": "inline-block", "width": "48%"})
    ]),

    html.Div([
        html.Label("Features for Training:"),
        dcc.Checklist(
            id='feature-selection',
            options=[],
            style={"margin": "auto", "padding": "10px"}
        )
    ], style={"textAlign": "center", "margin": "20px 0"}),

    html.Div([
        html.Button("Train Model", id='train-btn', n_clicks=0, style={"margin": "10px"}),
        html.Div(id='train-output', style={"textAlign": "center", "marginTop": "10px"})
    ], style={"textAlign": "center", "margin": "20px 0"}),

    html.Div([
        html.Label("Enter feature values (comma-separated):"),
        dcc.Input(
            id='predict-input',
            type='text',
            placeholder='Value1, Value2, ...',
            style={"width": "60%", "margin": "10px auto"}
        ),
        html.Button("Predict", id='predict-btn', n_clicks=0, style={"margin": "10px"}),
        html.Div(id='predict-output', style={"textAlign": "center", "marginTop": "10px"})
    ])
])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
def handleUpload(contents, filename):
    global dfG
    if contents is not None:
        contentType, contentString = contents.split(',')
        decoded = base64.b64decode(contentString)
        dfG = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        numCols = dfG.select_dtypes(include=['float64', 'int64']).columns
        dfG[numCols] = dfG[numCols].fillna(dfG[numCols].mean())
        return html.Div([html.H5(f"File uploaded: {filename}")])
    return html.Div("No file uploaded.")

@app.callback(
    Output('target-variable-dropdown', 'options'),
    Input('output-data-upload', 'children'))
def populateTargetDropdown(uploadStatus):
    global dfG
    if dfG is not None:
        numCols = dfG.select_dtypes(include=['float64', 'int64']).columns
        return [{"label": col, "value": col} for col in numCols]
    return []

@app.callback(
    Output('categorical-variable-radio', 'options'),
    Input('output-data-upload', 'children'))
def populateCatVars(uploadStatus):
    global dfG
    if dfG is not None:
        catCols = dfG.select_dtypes(include=['object']).columns.tolist()
        return [{"label": col, "value": col} for col in catCols]
    return []

@app.callback(
    [Output('avg-value-chart', 'figure'),
     Output('correlation-chart', 'figure')],
    [Input('target-variable-dropdown', 'value'),
     Input('categorical-variable-radio', 'value')])
def updateCharts(targetVar, catVar):
    global dfG
    if dfG is not None and targetVar is not None:
        if catVar:
            avgChart = px.bar(
                dfG.groupby(catVar)[targetVar].mean().reset_index(),
                x=catVar, y=targetVar,
                title=f"Average {targetVar} by {catVar}"
            )
        else:
            avgChart = px.bar(title="Select a categorical variable")

        numDf = dfG.select_dtypes(include=['float64', 'int64'])
        corr = numDf.corr()[targetVar].abs().sort_values(ascending=False).reset_index()
        corrChart = px.bar(
            corr.iloc[1:], x='index', y=targetVar,
            title=f"Correlation Strength of Numerical Variables with {targetVar}",
            labels={'index': 'Numerical Variables', targetVar: 'Correlation Strength'}
        )

        return avgChart, corrChart

    return px.bar(title="No data available"), px.bar(title="No data available")

@app.callback(
    Output('feature-selection', 'options'),
    Input('output-data-upload', 'children'))
def populateFeatures(uploadStatus):
    global dfG
    if dfG is not None:
        features = dfG.columns
        return [{"label": col, "value": col} for col in features]
    return []

@app.callback(
    Output('train-output', 'children'),
    [Input('train-btn', 'n_clicks')],
    [State('target-variable-dropdown', 'value'),
     State('feature-selection', 'value')])
def trainModel(nClicks, targetVar, selectedFeats):
    global dfG, model, featOrder
    if nClicks > 0 and targetVar is not None and selectedFeats:
        X = dfG[selectedFeats]
        y = dfG[targetVar]
        featOrder = selectedFeats

        numFeats = X.select_dtypes(include=['float64', 'int64']).columns
        catFeats = X.select_dtypes(include=['object']).columns

        numTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        catTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numTransformer, numFeats),
                ('cat', catTransformer, catFeats)
            ]
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(XTrain, yTrain)

        r2Score = model.score(XTest, yTest)

        return f"Model trained successfully! R² Score: {r2Score:.2f}"
    return " "

@app.callback(
    Output('predict-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('predict-input', 'value')])
def predict(nClicks, inputVals):
    global model, featOrder
    if nClicks > 0:
        if model is None:
            return "Model is not trained yet. Please train the model first."

        try:
            inputVals = inputVals.split(',')
            if len(inputVals) != len(featOrder):
                return f"Invalid, {len(featOrder)} values expected."

            inputDict = {featOrder[i]: inputVals[i].strip() for i in range(len(featOrder))}
            inputDf = pd.DataFrame([inputDict])

            for col in inputDf.columns:
                if col in dfG.select_dtypes(include=['float64', 'int64']).columns:
                    inputDf[col] = pd.to_numeric(inputDf[col], errors='coerce')

            prediction = model.predict(inputDf)

            return f"Predicted Target Value is: {prediction[0]:.2f}"
        except ValueError as e:
            return f"Error: {str(e)}. Ensure that input matches feature types and format."
        except Exception as e:
            return f"Unexpected Error: {str(e)}."
    return ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

server = app.server

