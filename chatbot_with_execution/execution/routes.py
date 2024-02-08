# chatbot_with_execution/execution/routes.py
from flask import render_template, request, jsonify
from chatbot_with_execution import app
from pylint.lint import Run
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Dummy dataset for illustration
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# chatbot_with_execution/execution/routes.py
from flask import render_template, request, jsonify
from chatbot_with_execution import app
from pylint.lint import Run
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Dummy dataset for illustration
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train a machine learning model and return evaluation metrics.
    """
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}

    except Exception as e:
        return {'error': str(e)}

def execute_code(code):
    try:
        exec_result = {}
        # Make the train_model function available for execution within provided code
        exec(code, {'train_model': train_model}, exec_result)
        return {'output': exec_result}

    except Exception as e:
        return {'error': str(e)}

def analyze_code(code):
    pylint_result = Run(
        ['--exit-zero', '--score=n', '--msg-template={path}:{line}:{column}: {msg} ({symbol})', '-'],
        do_exit=False, stdout=None, stderr=None, script='').lint(['-'])
    
    if pylint_result:
        return {'pylint_messages': pylint_result}
    else:
        return {'pylint_messages': [{'path': 'No issues found.', 'line': 0, 'column': 0, 'msg': 'No issues found.', 'symbol': 'C'}]}

def execute_and_analyze_code(code):
    execution_result = execute_code(code)
    analysis_result = analyze_code(code)
    return {**execution_result, **analysis_result}

@app.route('/execute', methods=['POST'])
def execute_and_analyze_endpoint():
    try:
        code = request.form.get('code')

        if code:
            result = execute_and_analyze_code(code)
            return jsonify(result)

        return jsonify({'error': 'No code provided for execution.'}), 400

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/execution/train_model', methods=['POST'])
def train_model_endpoint():
    try:
        model = RandomForestClassifier()  # You can replace this with your preferred model
        result = train_model(model, X_train, y_train, X_test, y_test)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'An error occurred during model training: {str(e)}'}), 500
