from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime

app = Flask(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["30 per hour"]
)

documents = None
doc_embeddings = None

def load_documents():
    """Load documents and their pre-computed embeddings"""
    global documents, doc_embeddings
    if documents is None:
        print("📚 Loading documents...")
        with open("hospital_docs.json") as f:
            documents = json.load(f)
        
        try:
            with open("hospital_embeddings.json") as f:
                doc_embeddings = json.load(f)
            print(f"✅ Loaded {len(documents)} documents with embeddings")
        except FileNotFoundError:
            print("⚠️ No embeddings file found, using keyword search")
            doc_embeddings = None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)

def get_embedding(text):
    """Get query embedding using gemini-embedding-001"""
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",   # ✅ Updated model
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def semantic_search(query, k=5):
    """Search using embeddings with cosine similarity"""
    load_documents()
    
    query_emb = get_embedding(query)
    if query_emb is None or doc_embeddings is None:
        return keyword_search(query, k)
    
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        if doc_emb is None:
            continue
        sim = cosine_similarity(query_emb, doc_emb)
        similarities.append((sim, i))
    
    similarities.sort(reverse=True)
    return [documents[i] for _, i in similarities[:k]]

def keyword_search(query, k=10):
    """
    Enhanced keyword search with medical synonym expansion.
    Used as fallback when embeddings are unavailable.
    """
    load_documents()

    # Medical synonym map — expands patient terms to document terms
    synonym_map = {
        "fracture": ["fracture", "broken", "bone", "orthopedic", "ortho"],
        "hair fracture": ["fracture", "hairline", "bone", "orthopedic"],
        "hairline": ["fracture", "hairline", "bone", "orthopedic"],
        "broken bone": ["fracture", "bone", "orthopedic"],
        "joint": ["joint", "orthopedic", "knee", "hip"],
        "heart": ["heart", "cardiac", "cardiology", "chest"],
        "chest pain": ["heart", "cardiac", "cardiology", "chest"],
        "child": ["child", "pediatric", "kid", "infant"],
        "baby": ["child", "pediatric", "infant", "pediatrics"],
        "pregnancy": ["pregnancy", "gynecology", "obstetrics", "maternity"],
        "women": ["gynecology", "obstetrics", "women"],
        "fever": ["general medicine", "fever", "infection"],
        "emergency": ["emergency", "24/7", "urgent"],
        "pooja": ["pooja", "emergency", "dr. pooja"],
        "xray": ["radiology", "x-ray", "imaging", "scan"],
        "x-ray": ["radiology", "x-ray", "imaging"],
    }

    query_lower = query.lower()
    expanded_keywords = set(query_lower.split())

    # Expand with synonyms
    for term, synonyms in synonym_map.items():
        if term in query_lower:
            expanded_keywords.update(synonyms)

    scored = []
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        score = sum(doc_lower.count(kw) for kw in expanded_keywords)
        if query_lower in doc_lower:
            score += 100
        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True)
    return [documents[i] for _, i in scored[:k]]

def generate_answer(query):
    """Generate answer using semantic search + Gemma 3"""
    load_documents()

    if doc_embeddings:
        relevant_docs = semantic_search(query, k=5)
    else:
        relevant_docs = keyword_search(query, k=5)

    if not relevant_docs:
        return "I don't have information about that. Please ask about doctors, facilities, or hospital services."

    context = '\n\n---\n\n'.join(relevant_docs)

    prompt = f"""You are the AI assistant for Sunrise Community Hospital. Help patients find the right doctor or service.

CRITICAL INSTRUCTIONS:
- Read the patient's symptoms or condition carefully
- Map them to the correct department using medical knowledge:
  * Fracture, broken bone, hairline fracture, bone pain, joint pain → ORTHOPEDICS (Dr. Karthik Menon)
  * Heart, chest pain, palpitations → CARDIOLOGY (Dr. Suresh Nair)
  * Child health, baby, infant → PEDIATRICS (Dr. Sunita Reddy / Dr. Vikram Singh)
  * Pregnancy, women's health → GYNECOLOGY (Dr. Anjali Desai / Dr. Meera Iyer)
  * Accident, emergency, trauma → EMERGENCY (Dr. Pooja Gupta / Dr. Arun Malhotra, available 24/7)
  * General illness, fever → GENERAL MEDICINE (Dr. Rajesh Kumar / Dr. Priya Sharma)
- Always include: doctor name, OPD days, and timings
- If an appointment link exists, include it
- Be concise and helpful

HOSPITAL INFORMATION:
{context}

PATIENT QUESTION: {query}

YOUR ANSWER:"""

    try:
        model = genai.GenerativeModel('models/gemma-3-4b-it')   # ✅ Updated model
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "I'm having trouble processing your request. Please try again in a moment."

@app.route('/health')
def health():
    return jsonify({
        'status': 'alive',
        'docs_loaded': documents is not None,
        'embeddings_loaded': doc_embeddings is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/ping')
def ping():
    return 'pong', 200

@app.route('/')
def home():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
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
                height: 100dvh;
                min-height: -webkit-fill-available;
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
                height: 90dvh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

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
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .message.user { justify-content: flex-end; }

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

            .message.user .message-icon { background: var(--light-blue); }

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

            .typing-indicator.active { display: flex; }

            .typing-dot {
                width: 8px;
                height: 8px;
                margin: 0 2px;
                background: var(--baby-blue);
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }

            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .typing-dot:nth-child(3) { animation-delay: 0.4s; }

            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }

            .chat-input-area {
                padding: 20px;
                background: var(--white);
                border-top: 2px solid var(--light-blue);
            }

            .chat-input-container { display: flex; gap: 10px; }

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

            #sendBtn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px var(--shadow); }
            #sendBtn:active { transform: translateY(0); }
            #sendBtn:disabled { opacity: 0.6; cursor: not-allowed; }

            .chat-messages::-webkit-scrollbar { width: 6px; }
            .chat-messages::-webkit-scrollbar-track { background: var(--off-white); }
            .chat-messages::-webkit-scrollbar-thumb { background: var(--baby-blue); border-radius: 10px; }

            @media (max-width: 768px) {
                .chat-container { height: 100dvh; border-radius: 0; max-width: 100%; }
                .chat-header h1 { font-size: 1.2rem; }
                .message-bubble { max-width: 85%; font-size: 0.9rem; }
                .chat-input-area { padding: 15px; }
                #sendBtn { padding: 12px 20px; }
            }

            @media (max-width: 480px) {
                .message-bubble { max-width: 90%; }
                .message-icon { width: 30px; height: 30px; font-size: 1rem; }
                #userInput { font-size: 0.9rem; }
                #sendBtn { font-size: 0.9rem; padding: 12px 18px; }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>✚ Sunrise Hospital</h1>
                <p>AI Assistant - Ask about doctors, timings & facilities</p>
            </div>

            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    <div class="message-icon">👩🏻‍⚕️</div>
                    <div class="message-bubble">
                        Hello! I\'m your hospital assistant. Ask me about which doctors to visit, their schedule, admission requirements, or any hospital facilities.
                    </div>
                </div>
            </div>

            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>

            <div class="chat-input-area">
                <div class="chat-input-container">
                    <input type="text" id="userInput" placeholder="Type your problem here..." autocomplete="off">
                    <button id="sendBtn">Send</button>
                </div>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById(\'chatMessages\');
            const userInput = document.getElementById(\'userInput\');
            const sendBtn = document.getElementById(\'sendBtn\');
            const typingIndicator = document.getElementById(\'typingIndicator\');

            function addMessage(text, isUser = false) {
                const messageDiv = document.createElement(\'div\');
                messageDiv.className = `message ${isUser ? \'user\' : \'bot\'}`;

                const icon = document.createElement(\'div\');
                icon.className = \'message-icon\';
                icon.textContent = isUser ? \'👤\' : \'👩🏻\u200d⚕️\';

                const bubble = document.createElement(\'div\');
                bubble.className = \'message-bubble\';

                const urlRegex = /(https?:\/\/[^\s]+)/g;
                const linkedText = text.replace(urlRegex, (url) => {
                    return `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color: #1596AF; text-decoration: underline;">${url}</a>`;
                });

                bubble.innerHTML = linkedText;

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

            function showTyping(show) {
                typingIndicator.classList.toggle(\'active\', show);
                if (show) chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            async function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;

                addMessage(message, true);
                userInput.value = \'\';
                sendBtn.disabled = true;
                showTyping(true);

                try {
                    const response = await fetch(\'/api/chat\', {
                        method: \'POST\',
                        headers: { \'Content-Type\': \'application/json\' },
                        body: JSON.stringify({ question: message })
                    });

                    const data = await response.json();
                    showTyping(false);
                    addMessage(data.answer, false);
                } catch (error) {
                    showTyping(false);
                    addMessage("Sorry, I\'m having trouble connecting. Please try again.", false);
                }

                sendBtn.disabled = false;
                userInput.focus();
            }

            sendBtn.addEventListener(\'click\', sendMessage);
            userInput.addEventListener(\'keypress\', (e) => {
                if (e.key === \'Enter\') sendMessage();
            });

            userInput.focus();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid request'}), 400

        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        print(f"📥 Question: {question}")
        answer = generate_answer(question)
        print(f"📤 Answer: {answer[:100]}...")

        return jsonify({'answer': answer}), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error. Please try again.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
