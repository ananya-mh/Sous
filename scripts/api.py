from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from transformers import pipeline
import pymysql
from pymysql import Error
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import re
import os
import json
from typing import Optional
from dotenv import load_dotenv

from grocery_list import add_diet_notes, apply_substitutions, build_grocery_list, flag_for_substitution, simple_parse
from recipe_search import (DEFAULT_GROQ_MODEL, QueryConstraints, groq_completer, load_or_build_embeddings, load_recipes, parse_query, search,
                           sentence_encoder, unsupported_diet)

load_dotenv()

app = FastAPI(title="Recipe AI API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days
# Configure password context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

security = HTTPBearer()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'recipeai'),
    'user': os.getenv('DB_USER', 'hari'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# If password is empty and not set via environment variable, show a warning
if not DB_CONFIG['password'] and not os.getenv('DB_PASSWORD'):
    print("\n⚠️  WARNING: MySQL password not configured!")
    print("   Please either:")
    print("   1. Set DB_PASSWORD environment variable, OR")
    print("   2. Modify DB_CONFIG['password'] in api.py")
    print("   Example: DB_CONFIG['password'] = 'your_mysql_password'")

# Initialize database connection
def get_db_connection():
    """Create and return database connection."""
    try:
        connection = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            port=DB_CONFIG['port'],
            cursorclass=pymysql.cursors.DictCursor  # pyright: ignore[reportAttributeAccessIssue]
        )
        return connection
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

# Initialize database tables
def init_db():
    """Initialize database tables."""
    connection = get_db_connection()
    if connection is None:
        print("⚠️ Could not connect to database. Authentication will not work.")
        return
    
    try:
        cursor = connection.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        print("✅ Database tables initialized successfully")
    except Error as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Initialize database on startup
init_db()

# Request/Response Models
class ParseRequest(BaseModel):
    text: str

class SubstituteRequest(BaseModel):
    item: str
    amount: str = ""
    unit: str = ""
    constraint: str

class AuthRequest(BaseModel):
    email: EmailStr
    password: str

class RecipeSearchRequest(BaseModel):
    query: str

class PastedRecipe(BaseModel):
    name: str = "My recipe"
    text: str = ""

class GroceryListRequest(BaseModel):
    recipe_ids: list[int] = []
    constraints: QueryConstraints = QueryConstraints()
    pasted_recipe: Optional[PastedRecipe] = None

# Configuration
MODEL_PATH = "bert_recipe_model"
BERT_MODEL_ID = os.getenv("BERT_MODEL_ID")  # optional Hugging Face Hub id, tried before MODEL_PATH

# Initialize models
print("⏳ Loading BERT Model...")
nlp = None
for model_source in [s for s in (BERT_MODEL_ID, MODEL_PATH) if s]:
    try:
        nlp = pipeline("token-classification", model=model_source, aggregation_strategy="first")
        print(f"✅ BERT Model loaded successfully from {model_source}")
        break
    except Exception as e:
        print(f"❌ Error loading BERT from {model_source}: {e}")

# Groq-hosted open-weight model for query parsing and substitutions; None (fallbacks only) without a key
ask_llm = groq_completer(os.getenv("GROQ_API_KEY"), os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))

# Recipe search data (optional: the search endpoints return 503 without RECIPES_CSV)
RECIPES_CSV = os.getenv("RECIPES_CSV")
recipes, recipes_by_id, recipe_embeddings, encode_query = [], {}, None, None
if RECIPES_CSV:
    print("⏳ Loading recipes...")
    try:
        recipes = load_recipes(RECIPES_CSV, int(os.getenv("RECIPES_LIMIT", "5000")),
                               cache_path=os.getenv("RECIPES_SUBSET_CACHE", "data/recipes_subset.json"))
        encode_query = sentence_encoder()
        recipe_embeddings = load_or_build_embeddings(
            recipes, os.getenv("RECIPES_EMBEDDINGS_CACHE", "data/recipe_embeddings.npz"), encode_query)
        recipes_by_id = {r["id"]: r for r in recipes}
        print(f"✅ Loaded {len(recipes)} recipes")
    except Exception as e:
        print(f"❌ Error loading recipes: {e}")
        recipes, recipes_by_id, recipe_embeddings = [], {}, None
else:
    print("⚠️ RECIPES_CSV not set: recipe search is disabled")

# Authentication utilities
def verify_password(plain_password, hashed_password):
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password."""
    # Bcrypt has a 72-byte limit, but passlib handles this automatically
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):  # pyright: ignore[reportArgumentType]
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")  # pyright: ignore[reportAssignmentType]
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, email FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user is None:
            raise credentials_exception
        return {"id": user['id'], "email": user['email']}  # pyright: ignore[reportCallIssue, reportArgumentType]
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def clean_text(text):
    """Fixes BERT tokenizer artifacts (## and punctuation spacing)."""
    text = text.replace("##", "")
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return text.strip()

def parse_recipe_bert(text):
    """Uses local BERT model to extract Amount, Unit, Item, Descriptor."""
    if nlp is None:
        raise Exception("BERT model not loaded")
    
    results = nlp(text)
    parsed = {"amount": [], "unit": [], "item": [], "descriptor": []}
    
    for entity in results:
        label = entity['entity_group']
        word = clean_text(entity['word'])
        
        if label == 'AMT': 
            parsed['amount'].append(word)
        elif label == 'UNIT': 
            parsed['unit'].append(word)
        elif label == 'NAME': 
            parsed['item'].append(word)
        elif label == 'DESC': 
            parsed['descriptor'].append(word)
            
    return {k: " ".join(v) if v else "" for k, v in parsed.items()}

def get_substitute(item, amount, unit, constraint):
    """Uses the LLM to find a substitute and calculate new math."""
    prompt = f"""
    Act as a professional food scientist.
    
    Task: Suggest a substitute for the ingredient below based on the constraint.
    
    Input: {amount} {unit} of {item}
    Constraint: {constraint}
    
    Rules:
    1. If a valid substitute exists, calculate the new quantity based on potency/density.
    2. If NO valid substitute exists for this constraint, return "found": false.
    3. Return ONLY valid JSON.
    
    Output JSON Format:
    {{
        "found": true,
        "substitute_item": "Name of new ingredient",
        "new_amount": "Number (decimal or fraction string)",
        "new_unit": "Unit (usually same, but changes for eggs/etc)",
        "reason": "Brief explanation of why this works"
    }}
    """
    
    try:
        if ask_llm is None:
            raise RuntimeError("GROQ_API_KEY is not set")
        clean_json = ask_llm(prompt).replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        return {"found": False, "reason": f"API Error: {str(e)}"}

# Authentication endpoints
@app.post("/api/auth/register")
async def register(request: AuthRequest):
    """Register a new user."""
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        cursor = connection.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (request.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        password_hash = get_password_hash(request.password)
        cursor.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
            (request.email, password_hash)
        )
        connection.commit()
        
        # Create access token
        access_token = create_access_token(data={"sub": request.email})
        
        return {
            "token": access_token,
            "user": {"id": cursor.lastrowid, "email": request.email}
        }
    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if connection:
            cursor.close()
            connection.close()

@app.post("/api/auth/login")
async def login(request: AuthRequest):
    """Login and get access token."""
    connection = get_db_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (request.email,))
        user = cursor.fetchone()
        
        if not user or not verify_password(request.password, user['password_hash']):  # pyright: ignore[reportCallIssue, reportArgumentType]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": user['email']})  # pyright: ignore[reportCallIssue, reportArgumentType]
        
        return {
            "token": access_token,
            "user": {"id": user['id'], "email": user['email']}  # pyright: ignore[reportCallIssue, reportArgumentType]
        }
    except HTTPException:
        raise
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Protected endpoints
@app.post("/api/parse")
async def parse(request: ParseRequest, current_user: dict = Depends(get_current_user)):
    """Parse recipe ingredient line."""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        parsed_result = parse_recipe_bert(request.text)
        return parsed_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/substitute")
async def substitute(request: SubstituteRequest, current_user: dict = Depends(get_current_user)):
    """Get ingredient substitute."""
    try:
        if not request.item or not request.constraint:
            raise HTTPException(status_code=400, detail="Item and constraint are required")
        
        substitute_result = get_substitute(
            request.item, 
            request.amount, 
            request.unit, 
            request.constraint
        )
        return substitute_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def require_recipes():
    if recipe_embeddings is None:
        raise HTTPException(status_code=503, detail="Recipe search is unavailable: set RECIPES_CSV and restart the server")

def pasted_lines(text):
    """One ingredient per line; drops blank lines and leading bullets ("-", "*", "•")."""
    lines = (re.sub(r"^\s*[-*•]+\s*", "", line).strip() for line in text.splitlines())
    return [line for line in lines if line]

# Plain `def`: FastAPI runs these in a worker thread, so blocking LLM/encoder calls don't stall the server.
@app.post("/api/recipes/search")
def search_recipes(request: RecipeSearchRequest, current_user: dict = Depends(get_current_user)):
    """Parse a plain-language request and return the top 5 matching recipes."""
    require_recipes()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    constraints, parser = parse_query(request.query, ask_llm)
    results = search(constraints, request.query, recipes, recipe_embeddings, encode_query)
    return {
        "constraints": constraints.model_dump(),
        "parser": parser,  # "llm", or "fallback" when LLM parsing failed
        "unsupported_diet": unsupported_diet(constraints),
        "results": [{
            "id": r["recipe"]["id"],
            "name": r["recipe"]["name"],
            "minutes": r["recipe"]["minutes"],
            "top_ingredients": r["recipe"]["ingredients"][:6],
            "ingredient_count": len(r["recipe"]["ingredients"]),
            "tags": r["recipe"]["tags"][:6],
            "missing_ingredients": r["missing_ingredients"],
            "diet_notes": r["diet_notes"],
            "score": round(r["score"], 4),
        } for r in results],
    }

@app.get("/api/recipes/{recipe_id}")
def get_recipe(recipe_id: int, current_user: dict = Depends(get_current_user)):
    """Full recipe for the detail view: ingredients with amount hints (Food.com has no units) and steps."""
    recipe = recipes_by_id.get(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"Unknown recipe id: {recipe_id}")
    hints = recipe["amount_hints"] or [None] * len(recipe["ingredients"])
    return {
        "id": recipe["id"],
        "name": recipe["name"],
        "minutes": recipe["minutes"],
        "tags": recipe["tags"],
        "ingredients": [{"name": name, "amount": amount} for name, amount in zip(recipe["ingredients"], hints)],
        "steps": recipe["steps"],
    }

@app.post("/api/recipes/grocery-list")
def grocery_list(request: GroceryListRequest, current_user: dict = Depends(get_current_user)):
    """Combined grocery list for chosen Food.com recipes plus an optional pasted recipe,
    with substitutions for items that break the constraints."""
    pasted = pasted_lines(request.pasted_recipe.text) if request.pasted_recipe else []
    if not request.recipe_ids and not pasted:
        raise HTTPException(status_code=400, detail="Choose at least one recipe or paste one")
    if request.recipe_ids:
        require_recipes()
    unknown = [i for i in request.recipe_ids if i not in recipes_by_id]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown recipe ids: {unknown}")

    # Food.com recipes have no units, so they merge by name (amount hints only); pasted lines are parsed.
    recipe_inputs = [{"name": recipes_by_id[i]["name"], "lines": None,
                      "ingredient_names": recipes_by_id[i]["ingredients"],
                      "amount_hints": recipes_by_id[i]["amount_hints"]} for i in request.recipe_ids]
    if pasted:
        recipe_inputs.append({"name": request.pasted_recipe.name.strip() or "My recipe", "lines": pasted})

    parse_fn = parse_recipe_bert if nlp is not None else simple_parse
    result = build_grocery_list(recipe_inputs, parse_fn)
    constraints = request.constraints
    flagged = flag_for_substitution(result["items"] + result["unmerged"],
                                    diet=constraints.diet, exclude=constraints.exclude_ingredients)
    substitutions = apply_substitutions(flagged, get_substitute)
    add_diet_notes(result["items"] + result["unmerged"], constraints.diet)
    return {**result, "substitutions": substitutions, "parser": "bert" if nlp is not None else "fallback"}

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "bert_loaded": nlp is not None,
        "recipes_loaded": len(recipes),
        "db_connected": get_db_connection() is not None
    }

if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*50)
    print(" 🥗 Recipe AI API Server (FastAPI)")
    print("="*50)
    print("Starting server on http://localhost:5000")
    print("API docs available at http://localhost:5000/docs")
    print(f"\nDatabase Configuration:")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Port: {DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  User: {DB_CONFIG['user']}")
    print(f"  Password: {'*' * len(DB_CONFIG['password']) if DB_CONFIG['password'] else '(not set - connection may fail)'}")
    uvicorn.run(app, host="0.0.0.0", port=5000)
