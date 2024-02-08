# chatbot_with_execution/chat/models.py
from chatbot_with_execution import db

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(255))
    bot = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
