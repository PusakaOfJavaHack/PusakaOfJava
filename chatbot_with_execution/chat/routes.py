# chatbot_with_execution/chat/routes.py
from flask import render_template, jsonify
from flask_socketio import emit
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatbot_with_execution import socketio, db, nlp
from chatbot_with_execution.chat.models import Conversation
from datetime import datetime

chatbot = ChatBot('MyBot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')

@socketio.on('user_input', namespace='/chat')
def handle_user_input(message):
    try:
        user_message = message.get('user_message', '').strip()

        if not user_message:
            emit('bot_response', {'bot_response': "Please enter a valid message."})
            return

        doc = nlp(user_message)
        entities = ', '.join([ent.text for ent in doc.ents])

        bot_response = str(chatbot.get_response(user_message))
        if entities:
            bot_response += f" Entities: {entities}"

        conversation = Conversation(user=user_message, bot=bot_response, timestamp=datetime.now())
        db.session.add(conversation)
        db.session.commit()

        emit('bot_response', {'bot_response': bot_response, 'timestamp': conversation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}, broadcast=True)
        trainer.train([user_message, bot_response])

    except Exception as e:
        print(f"Error: {str(e)}")
        emit('bot_response', {'bot_response': "An error occurred. Please try again later."})

@chat_blueprint.route('/')
def home():
    try:
        conversation_history = Conversation.query.order_by(Conversation.timestamp.desc()).limit(10).all()
        return render_template('chat/index.html', conversation=conversation_history)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving conversation history.'}), 500
