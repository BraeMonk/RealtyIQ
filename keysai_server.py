import os
import re
import json
import math
import time
import sqlite3
import requests

from datetime import datetime
from collections import defaultdict

from flask import Flask, jsonify, request
from flask_cors import CORS

from rapidfuzz import fuzz

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

APP_NAME = "KeysAI Ontario Intelligence"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL_FAST = "phi3"
OLLAMA_MODEL_REASONING = "mistral"

PORT = 5000

DATA_DIR = "data"

MAX_CONTEXT_DOCS = 10

# ═══════════════════════════════════════════════════════════════
# FLASK
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════

DB_PATH = "keysai.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    message TEXT,
    response TEXT,
    created_at TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_memory (
    user_id TEXT,
    key TEXT,
    value TEXT
)
""")

conn.commit()

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_json(filename):

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

market_data = load_json("ontario_markets.json")
listing_data = load_json("listings.json")
neighbourhood_data = load_json("neighbourhoods.json")
mortgage_data = load_json("mortgage_rules.json")
program_data = load_json("buyer_programs.json")

# ═══════════════════════════════════════════════════════════════
# TRUST SCORING
# ═══════════════════════════════════════════════════════════════

SOURCE_TRUST = {
    "CREA": 10,
    "TRREB": 10,
    "CMHC": 10,
    "Bank of Canada": 10,
    "Statistics Canada": 10,
    "Ontario.ca": 10,
    "Municipal Data": 8,
    "Regional Real Estate Board": 8,
    "Local Market Report": 6,
    "Editorial": 4,
    "Unknown": 1
}

# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def now():
    return datetime.utcnow().isoformat()

def tokenize(text):

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.split()

def format_price(n):

    if not n:
        return "—"

    if n >= 1_000_000:
        return f"${round(n/1_000_000,2)}M"

    return f"${n:,.0f}"

# ═══════════════════════════════════════════════════════════════
# MEMORY ENGINE
# ═══════════════════════════════════════════════════════════════

def save_memory(user_id, key, value):

    cursor.execute("""
    INSERT INTO user_memory (user_id,key,value)
    VALUES (?,?,?)
    """, (user_id, key, value))

    conn.commit()

def get_memory(user_id):

    cursor.execute("""
    SELECT key,value FROM user_memory
    WHERE user_id = ?
    """, (user_id,))

    rows = cursor.fetchall()

    memory = {}

    for row in rows:
        memory[row[0]] = row[1]

    return memory

# ═══════════════════════════════════════════════════════════════
# INTENT ENGINE
# ═══════════════════════════════════════════════════════════════

def detect_intent(query):

    q = query.lower()

    if any(x in q for x in [
        "mortgage",
        "payment",
        "interest rate",
        "afford",
        "qualification"
    ]):
        return "mortgage"

    if any(x in q for x in [
        "investment",
        "cash flow",
        "cap rate",
        "rental"
    ]):
        return "investment"

    if any(x in q for x in [
        "market",
        "trend",
        "prices",
        "inventory"
    ]):
        return "market"

    if any(x in q for x in [
        "listing",
        "home",
        "condo",
        "detached",
        "townhouse"
    ]):
        return "listing"

    return "general"

# ═══════════════════════════════════════════════════════════════
# ENTITY EXTRACTION
# ═══════════════════════════════════════════════════════════════

ONTARIO_CITIES = [
    "Toronto",
    "Ottawa",
    "Hamilton",
    "London",
    "Kitchener",
    "Waterloo",
    "Cambridge",
    "Barrie",
    "Vaughan",
    "Markham",
    "Oakville",
    "Kingston",
    "Windsor",
    "Niagara",
    "Sudbury",
    "Thunder Bay"
]

def extract_city(query):

    q = query.lower()

    for city in ONTARIO_CITIES:

        if city.lower() in q:
            return city

    return None

def extract_budget(query):

    q = query.lower()

    match = re.search(r'(\$)?([\d,.]+)\s?(k|m)?', q)

    if not match:
        return None

    num = float(match.group(2).replace(",", ""))

    suffix = match.group(3)

    if suffix == "k":
        num *= 1000

    if suffix == "m":
        num *= 1_000_000

    return int(num)

# ═══════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL ENGINE
# ═══════════════════════════════════════════════════════════════

def score_document(query, doc):

    score = 0

    q = query.lower()

    content = doc.get("content", "").lower()

    tags = doc.get("tags", [])

    # semantic-ish fuzzy match
    score += fuzz.partial_ratio(q, content) * 0.45

    # city boost
    if doc.get("city"):

        if doc["city"].lower() in q:
            score += 40

    # tag matching
    for tag in tags:

        if tag.lower() in q:
            score += 25

    # trust weighting
    source = doc.get("source", "Unknown")

    trust = SOURCE_TRUST.get(source, 1)

    score += trust * 5

    return score

def retrieve_documents(query, limit=MAX_CONTEXT_DOCS):

    combined = []

    datasets = [
        market_data,
        listing_data,
        neighbourhood_data,
        mortgage_data,
        program_data
    ]

    for dataset in datasets:

        for doc in dataset:

            scored = dict(doc)

            scored["_score"] = score_document(query, doc)

            combined.append(scored)

    combined.sort(key=lambda x: x["_score"], reverse=True)

    return combined[:limit]

# ═══════════════════════════════════════════════════════════════
# LISTING SEARCH
# ═══════════════════════════════════════════════════════════════

def search_listings(query):

    city = extract_city(query)

    budget = extract_budget(query)

    results = []

    for listing in listing_data:

        if city:

            if listing.get("city") != city:
                continue

        if budget:

            if listing.get("price", 0) > budget:
                continue

        results.append(listing)

    return results[:20]

# ═══════════════════════════════════════════════════════════════
# PROPERTY SCORING ENGINE
# ═══════════════════════════════════════════════════════════════

def score_listing(listing):

    score = {
        "investment_score": 5,
        "family_score": 5,
        "commuter_score": 5,
        "luxury_score": 5
    }

    if listing.get("beds", 0) >= 3:
        score["family_score"] += 2

    if listing.get("parking", 0) >= 2:
        score["family_score"] += 1

    if listing.get("city") in ["Toronto", "Ottawa"]:
        score["investment_score"] += 2

    if listing.get("price", 0) >= 1_500_000:
        score["luxury_score"] += 3

    return score

# ═══════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_context(documents):

    chunks = []

    for doc in documents:

        chunk = f"""

TITLE:
{doc.get('title', 'Untitled')}

CITY:
{doc.get('city', 'Ontario')}

SOURCE:
{doc.get('source', 'Unknown')}

CONTENT:
{doc.get('content', '')}

"""

        chunks.append(chunk)

    return "\n".join(chunks)

# ═══════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ═══════════════════════════════════════════════════════════════

def execute_tools(intent, query):

    tools_output = {}

    if intent == "listing":

        listings = search_listings(query)

        enriched = []

        for listing in listings:

            item = dict(listing)

            item["ai_scores"] = score_listing(listing)

            enriched.append(item)

        tools_output["listings"] = enriched

    return tools_output

# ═══════════════════════════════════════════════════════════════
# SOURCE COLLECTION
# ═══════════════════════════════════════════════════════════════

def collect_sources(documents):

    sources = []

    for doc in documents:

        src = doc.get("source")

        if src and src not in sources:
            sources.append(src)

    return sources

# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are KeysAI.

You are an Ontario real estate intelligence system.

RULES:

1. Never fabricate facts.
2. Never invent listings.
3. Only use supplied context.
4. Prioritize trusted Canadian sources.
5. Be analytical and concise.
6. Explain reasoning clearly.
7. If uncertain, explicitly say so.
8. Be useful and professional.
9. Think like an expert advisor.
10. Include actionable next steps.
"""

# ═══════════════════════════════════════════════════════════════
# OLLAMA
# ═══════════════════════════════════════════════════════════════

def ask_ollama(model, prompt):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        data = response.json()

        return data.get("response", "").strip()

    except Exception as e:

        return f"Ollama error: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# REASONING PIPELINE
# ═══════════════════════════════════════════════════════════════

def generate_response(query, user_id="anonymous"):

    intent = detect_intent(query)

    documents = retrieve_documents(query)

    context = build_context(documents)

    tools = execute_tools(intent, query)

    memory = get_memory(user_id)

    sources = collect_sources(documents)

    planning_prompt = f"""
You are an Ontario real estate planner.

USER QUERY:
{query}

Determine:

1. User goal
2. Key concerns
3. Best response strategy
4. What tools matter most
"""

    plan = ask_ollama(
        OLLAMA_MODEL_FAST,
        planning_prompt
    )

    synthesis_prompt = f"""
{SYSTEM_PROMPT}

USER MEMORY:
{json.dumps(memory, indent=2)}

PLANNING:
{plan}

TOOLS:
{json.dumps(tools, indent=2)}

CONTEXT:
{context}

USER QUESTION:
{query}

Provide:

1. Direct answer
2. Supporting reasoning
3. Helpful guidance
4. Mention uncertainty if needed
5. Never hallucinate
"""

    final_answer = ask_ollama(
        OLLAMA_MODEL_REASONING,
        synthesis_prompt
    )

    return {
        "message": final_answer,
        "intent": intent,
        "sources": sources,
        "tools": tools
    }

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def root():

    return jsonify({
        "name": APP_NAME,
        "status": "running",
        "time": now()
    })

@app.route("/health")
def health():

    try:

        r = requests.get("http://localhost:11434")

        ollama_online = r.status_code == 200

    except:
        ollama_online = False

    return jsonify({
        "server": "ok",
        "ollama": ollama_online,
        "model_fast": OLLAMA_MODEL_FAST,
        "model_reasoning": OLLAMA_MODEL_REASONING
    })

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    message = data.get("message", "")
    user_id = data.get("user_id", "anonymous")

    if not message:

        return jsonify({
            "error": "No message"
        }), 400

    result = generate_response(message, user_id)

    cursor.execute("""
    INSERT INTO conversations
    (user_id,message,response,created_at)
    VALUES (?,?,?,?)
    """, (
        user_id,
        message,
        result["message"],
        now()
    ))

    conn.commit()

    return jsonify({
        "success": True,
        "response": result["message"],
        "intent": result["intent"],
        "sources": result["sources"],
        "tools": result["tools"]
    })

@app.route("/market/<city>")
def market(city):

    city = city.lower()

    for item in market_data:

        if item.get("city", "").lower() == city:

            return jsonify(item)

    return jsonify({
        "error": "City not found"
    }), 404

@app.route("/listings/search", methods=["POST"])
def listings_search():

    data = request.get_json()

    query = data.get("query", "")

    results = search_listings(query)

    enriched = []

    for listing in results:

        item = dict(listing)

        item["ai_scores"] = score_listing(listing)

        enriched.append(item)

    return jsonify({
        "count": len(enriched),
        "results": enriched
    })

# ═══════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n================================================")
    print(" KeysAI Ontario Intelligence Server")
    print(" Local AI + Ollama + RAG")
    print("================================================")
    print(f" Fast Model: {OLLAMA_MODEL_FAST}")
    print(f" Reasoning Model: {OLLAMA_MODEL_REASONING}")
    print(f" Port: {PORT}")
    print("================================================\n")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=True
    )
