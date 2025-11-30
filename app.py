from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os
import json
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import warnings
import psutil

 
warnings.filterwarnings("ignore")

app = Flask(__name__)

load_dotenv()
api_ky = os.getenv("GEMINI_API_KEY")

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20 per hour"]
)


embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Indexes")
index = faiss.read_index("hospital_index.faiss")
print("Documents chunks")
documents = json.load(open("hospital_docs.json"))

client = genai.Client(api_key=api_ky)


def retrieve_context(query, k=5):
    query_emb = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_emb), k)
    return [documents[i] for i in indices[0]]


def print_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)
    cpu = process.cpu_percent(interval=0.1)
    print(f"RAM Used: {mem:.2f} MB, CPU: {cpu:.1f}%")


def generate_answer(query):
    context = retrieve_context(query)

    prompt = f"""
You are a question-answering system.
Use ONLY the following context to answer.
If the answer is not present, reply exactly: "I don't know".

CONTEXT:
{ ' '.join(context) }

QUESTION:
{ query }

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()

@app.route('/')
def home():
    html_content = '''
    <<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sunrise Hospital - AI-RAG Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                --baby-blue: #1596AF;
                --light-blue: #bcebf4ff;
                --white: #FFFFFF;
                --off-white: #F8FCFF;
                --dark-blue: #2cb1ccff;
                --text-dark: #2C3E50;
                --text-light: #5A6C7D;
                --shadow: rgba(137, 207, 240, 0.2);
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background: linear-gradient(135deg, var(--off-white) 0%, var(--light-blue) 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 10px;
            }

            .chat-container {
                background: var(--white);
                border-radius: 20px;
                box-shadow: 0 10px 40px var(--shadow);
                width: 100%;
                max-width: 800px;
                height: 90vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            /* Header */
            .chat-header {
                background: linear-gradient(135deg, var(--baby-blue) 0%, var(--dark-blue) 100%);
                color: var(--white);
                padding: 20px;
                text-align: left;
                box-shadow: 0 2px 10px var(--shadow);
            }

            .chat-header h1 {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 5px;
            }

            .chat-header p {
                font-size: 0.9rem;
                opacity: 0.9;
            }

            /* Messages Area */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: var(--off-white);
            }

            .message {
                display: flex;
                margin-bottom: 15px;
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .message.user {
                justify-content: flex-end;
            }

            .message-bubble {
                max-width: 75%;
                padding: 12px 18px;
                border-radius: 18px;
                font-size: 0.95rem;
                line-height: 1.5;
                word-wrap: break-word;
            }

            .message.bot .message-bubble {
                background: var(--white);
                color: var(--text-dark);
                border: 2px solid var(--light-blue);
                border-radius: 18px 18px 18px 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }

            .message.user .message-bubble {
                background: linear-gradient(135deg, var(--baby-blue) 0%, var(--dark-blue) 100%);
                color: var(--white);
                border-radius: 18px 18px 4px 18px;
                box-shadow: 0 2px 8px var(--shadow);
            }

            .message-icon {
                width: 35px;
                height: 35px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
                margin: 0 10px;
                flex-shrink: 0;
            }

            .message.bot .message-icon {
                background: linear-gradient(135deg, var(--baby-blue) 0%, var(--dark-blue) 100%);
            }

            .message.user .message-icon {
                background: var(--light-blue);
            }

            /* Typing Indicator */
            .typing-indicator {
                display: none;
                align-items: center;
                padding: 12px 18px;
                background: var(--white);
                border: 2px solid var(--light-blue);
                border-radius: 18px;
                width: fit-content;
                margin-left: 45px;
            }

            .typing-indicator.active {
                display: flex;
            }

            .typing-dot {
                width: 8px;
                height: 8px;
                margin: 0 2px;
                background: var(--baby-blue);
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }

            .typing-dot:nth-child(2) {
                animation-delay: 0.2s;
            }

            .typing-dot:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-10px);
                }
            }

            /* Input Area */
            .chat-input-area {
                padding: 20px;
                background: var(--white);
                border-top: 2px solid var(--light-blue);
            }

            .chat-input-container {
                display: flex;
                gap: 10px;
            }

            #userInput {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid var(--light-blue);
                border-radius: 25px;
                font-size: 0.95rem;
                outline: none;
                transition: all 0.3s ease;
                background: var(--off-white);
            }

            #userInput:focus {
                border-color: var(--baby-blue);
                background: var(--white);
                box-shadow: 0 0 0 3px rgba(137, 207, 240, 0.1);
            }

            #sendBtn {
                padding: 14px 24px;
                background: linear-gradient(135deg, var(--baby-blue) 0%, var(--dark-blue) 100%);
                color: var(--white);
                border: none;
                border-radius: 25px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px var(--shadow);
            }

            #sendBtn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px var(--shadow);
            }

            #sendBtn:active {
                transform: translateY(0);
            }

            #sendBtn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            /* Scrollbar */
            .chat-messages::-webkit-scrollbar {
                width: 6px;
            }

            .chat-messages::-webkit-scrollbar-track {
                background: var(--off-white);
            }

            .chat-messages::-webkit-scrollbar-thumb {
                background: var(--baby-blue);
                border-radius: 10px;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .chat-container {
                    height: 100vh;
                    border-radius: 0;
                    max-width: 100%;
                }

                .chat-header h1 {
                    font-size: 1.2rem;
                }

                .message-bubble {
                    max-width: 85%;
                    font-size: 0.9rem;
                }

                .chat-input-area {
                    padding: 15px;
                }

                #sendBtn {
                    padding: 12px 20px;
                }
            }

            @media (max-width: 480px) {
                .message-bubble {
                    max-width: 90%;
                }

                .message-icon {
                    width: 30px;
                    height: 30px;
                    font-size: 1rem;
                }

                #userInput {
                    font-size: 0.9rem;
                }

                #sendBtn {
                    font-size: 0.9rem;
                    padding: 12px 18px;
                }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <!-- Header -->
            <div class="chat-header">
                <h1>✚ Sunrise Hospital</h1>
                <p>AI Assistant - Ask about doctors, timings & facilities</p>
            </div>

            <!-- Messages Area -->
            <div class="chat-messages" id="chatMessages">
                <!-- Welcome Message -->
                <div class="message bot">
                    <div class="message-icon">👩🏻‍⚕️</div>
                    <div class="message-bubble">
                        Hello! I'm your hospital assistant. Ask me about which doctors to visit there schedule, admission requirements, or any hospital facilities.
                    </div>
                </div>
            </div>

            <!-- Typing Indicator -->
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>

            <!-- Input Area -->
            <div class="chat-input-area">
                <div class="chat-input-container">
                    <input 
                        type="text" 
                        id="userInput" 
                        placeholder="Type your problem here..."
                        autocomplete="off"
                    >
                    <button id="sendBtn">Send</button>
                </div>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const userInput = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');
            const typingIndicator = document.getElementById('typingIndicator');

            // Add message to chat
            function addMessage(text, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

                const icon = document.createElement('div');
                icon.className = 'message-icon';
                icon.textContent = isUser ? '👤' : '🤖';

                const bubble = document.createElement('div');
                bubble.className = 'message-bubble';
                bubble.textContent = text;

                if (isUser) {
                    messageDiv.appendChild(bubble);
                    messageDiv.appendChild(icon);
                } else {
                    messageDiv.appendChild(icon);
                    messageDiv.appendChild(bubble);
                }

                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            // Show/hide typing indicator
            function showTyping(show) {
                typingIndicator.classList.toggle('active', show);
                if (show) {
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }

            // Send message
            async function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;

                // Add user message
                addMessage(message, true);
                userInput.value = '';
                sendBtn.disabled = true;

                // Show typing
                showTyping(true);

                try {
                    // Call your backend API here
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: message })
                    });

                    const data = await response.json();

                    // Hide typing and show response
                    showTyping(false);
                    addMessage(data.answer, false);

                } catch (error) {
                    showTyping(false);
                    addMessage("Sorry, I'm having trouble connecting. Please try again.", false);
                }

                sendBtn.disabled = false;
                userInput.focus();
            }

            // Event listeners
            sendBtn.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            // Focus input on load
            userInput.focus();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("5 per minute")
def chat():
    data = request.json
    question = data.get('question', '')
    print_usage()
        
    if not question:
        return jsonify({'error': 'No question provided'}),400
    try:
        answer = generate_answer(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    print("🚀 Starting Smart Healthcare Chatbot on Render...")
    port = int(os.environ.get("PORT", 10000))   # Render assigns this port
    app.run(host='0.0.0.0', port=port)


