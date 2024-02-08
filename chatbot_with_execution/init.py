# chatbot_with_execution/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from chatbot_with_execution.chat.routes import chat_blueprint
from chatbot_with_execution.execution.routes import execution_blueprint
from chatbot_with_execution.execution.routes import train_model
import spacy

app = Flask(__name__)
app.config.from_pyfile('.env')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app)

app.register_blueprint(chat_blueprint, url_prefix='/chat')
app.register_blueprint(execution_blueprint, url_prefix='/execution')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

if __name__ == '__main__':
    socketio.run(app, debug=True)
