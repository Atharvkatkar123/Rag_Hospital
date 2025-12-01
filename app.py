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
        
        # Try to load pre-computed embeddings
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
    """Get embedding from Google API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def semantic_search(query, k=5):
    """Search using embeddings"""
    load_documents()
    
    # Get query embedding
    query_emb = get_embedding(query)
    if query_emb is None or doc_embeddings is None:
        # Fallback to keyword search
        return keyword_search(query, k)
    
    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_emb, doc_emb)
        similarities.append((sim, i))
    
    # Sort by similarity and return top k
    similarities.sort(reverse=True)
    return [documents[i] for _, i in similarities[:k]]

def keyword_search(query, k=10):
    """Fallback keyword search"""
    load_documents()
    keywords = query.lower().split()
    scored = []
    
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        # Count keyword occurrences
        score = sum(doc_lower.count(kw) for kw in keywords)
        # Bonus for exact phrase match
        if query.lower() in doc_lower:
            score += 100
        if score > 0:
            scored.append((score, i))
    
    scored.sort(reverse=True)
    return [documents[i] for _, i in scored[:k]]

def generate_answer(query):
    """Generate answer using semantic search + Gemini"""
    # Use semantic search if embeddings available, else keyword
    if doc_embeddings:
        relevant_docs = semantic_search(query, k=5)
    else:
        relevant_docs = keyword_search(query, k=10)
    
    if not relevant_docs:
        return "I don't have information about that. Please ask about doctors, facilities, or hospital services."
    
    context = '\n\n'.join(relevant_docs)
    
    prompt = f"""You are Sunrise Community Hospital's AI assistant.

Answer the patient's question using ONLY the information below.
Be helpful, professional, and concise.
If the answer isn't in the context, say "I don't have that specific information."

HOSPITAL INFORMATION:
{context}

PATIENT QUESTION: {query}

YOUR ANSWER:"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
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
    html_content = '''<!DOCTYPE html>
    <!-- YOUR EXISTING HTML HERE -->
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
