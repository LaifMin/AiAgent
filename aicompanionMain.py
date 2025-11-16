import os
import yaml
import logging
import re
import asyncio
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore


try:
    from kokoro import KPipeline, KModel
    import soundfile as soundfile
    from pydub import AudioSegment
    import base64
    import whisper
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Audio features disabled. ")

 

# Load configuration file
with open('config.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})



# Load LLM model
logging.info(f"Loading AI model: {config['models']['ai']}")

# Model configuration for memory optimization
model_kwargs = {
    'model': config['models']['ai'],
    'temperature': 0,
    'reasoning': False
}

# Add memory optimization settings if available
if 'num_ctx' in config['models']:
    model_kwargs['num_ctx'] = config['models']['num_ctx']
    logging.info(f"Context window set to: {config['models']['num_ctx']}")
if 'num_gpu' in config['models']:
    model_kwargs['num_gpu'] = config['models']['num_gpu']
    logging.info(f"GPU layers set to: {config['models']['num_gpu']} (0 = CPU only)")

lcmodel = ChatOllama(**model_kwargs)

# Load embeddings and vector store for RAG
logging.info(f"Loading embeddings: {config['models']['embedding']}")
embeddings = OllamaEmbeddings(model=config['models']['embedding'])
vs = InMemoryVectorStore.load(config['rag_db_path'], embeddings)
retriever = vs.as_retriever()


# Define contexts for each character
contextOne = [
    ('system', config['prompts']['Alice']),
    ('system', '')
]

contextTwo = [
    ('system', config['prompts']['Arthur']),
    ('system', '')
]

contextThree = [
    ('system', config['prompts']['Ruby']),
    ('system', '')
]

# Dictionary to manage all character contexts
dictContexts = {
    'Alice': contextOne,
    'Arthur': contextTwo,
    'Ruby': contextThree
}

def get_other_characters_context(current_character):
    """
    Get the conversation history from other characters
    This allows a character to reference what others have discussed
    """
    other_contexts = []
    for char_name, char_context in dictContexts.items():
        if char_name != current_character:
            conversation = char_context[2:]  # Skip first 2 system messages
            if conversation:
                other_contexts.append({
                    'character': char_name,
                    'messages': conversation
                })
    return other_contexts

def format_other_contexts_for_prompt(other_contexts):
    """
    Format other characters' conversations for inclusion in prompt
    """
    if not other_contexts:
        return ""
    
    formatted = "\n\n--- CONVERSAZIONI DEGLI ALTRI PERSONAGGI (per tua informazione) ---\n"
    for ctx in other_contexts:
        formatted += f"\n{ctx['character']}:\n"
        for role, content in ctx['messages']:
            if role == 'human':
                formatted += f"  Utente: {content}\n"
            elif role == 'ai':
                formatted += f"  {ctx['character']}: {content}\n"
    formatted += "\n--- FINE CONVERSAZIONI ALTRI PERSONAGGI ---\n"
    return formatted



# Test model connection
logging.info("Testing model connection...")
try:
    lcmodel.invoke([('human', 'test')])
    logging.info("Model ready!")
except Exception as e:
    logging.error(f"Model initialization failed: {e}")



if HAS_AUDIO:
    # TTS Setup
    logging.info("Initializing Text-to-Speech...")
    kmodel = KModel(
        model=config['kokoro']['model'], 
        config=config['kokoro']['config']
    )
    kpipeline = KPipeline(lang_code='i', model=kmodel)
    
    # STT Setup
    logging.info("Loading Whisper model for Speech-to-Text...")
    wmodel = whisper.load_model(config['models']['whisper'], device="cpu")
    logging.info("Audio features enabled!")

def text_to_speech(text):
    """Convert text to speech and return base64 encoded audio"""
    if not HAS_AUDIO:
        return None
    
    try:
        # Ensure output folder exists
        os.makedirs('./generated_audio', exist_ok=True)
        
        # Clean text for better TTS
        text = text.replace("*", "")
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Generate audio chunks
        generator = kpipeline(text, voice=config['kokoro']['voice'], speed=0.9)
        audio = AudioSegment.empty()

        for i, (gs, ps, audio_chunk) in enumerate(generator):
            soundfile.write(
                f'./generated_audio/output_{i}.wav', 
                audio_chunk, 
                24000, 
                "PCM_16"
            )
            audio += AudioSegment.from_wav(f'./generated_audio/output_{i}.wav')

        audio.export('./generated_audio/final_output.wav', format='wav')

        # Read and encode to base64
        with open('./generated_audio/final_output.wav', 'rb') as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_b64
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return None

def speech_to_text(file_path):
    """Convert speech to text using Whisper"""
    if not HAS_AUDIO:
        return ""
    
    try:
        result = wmodel.transcribe(
            audio=file_path, 
            language="it", 
            fp16=False
        )
        return result.get("text", "")
    except Exception as e:
        logging.error(f"STT error: {e}")
        return ""



def is_safe(message):
    try:
        prompt = config['prompts']['secure'].format(message=message)
        response = lcmodel.invoke([('human', prompt)])
        is_safe = "unsafe" not in response.content.lower()
        
        if not is_safe:
            logging.warning(f"Unsafe request detected: {message[:100]}...")
        
        return is_safe
    except Exception as e:
        logging.error(f"Security check error: {e}")
        return True  # Default to safe if check fails



async def userRequest(user_message, character_name, check_safety=False, include_other_contexts=True):
    """
    Process user request for a specific character
    
    Args:
        user_message: The user's message
        character_name: Which character is responding ('Alice', 'Arthur', 'Ruby')
        check_safety: Whether to perform safety check
        include_other_contexts: Whether to include other characters' conversations
    """
    
    # Security check (optional)
    if check_safety and not is_safe(user_message):
        return "Mi dispiace, non posso rispondere a questa richiesta."
    
    logging.info(f"[{character_name}] User: {user_message}")
    
    # Get the context for this character
    context = dictContexts[character_name]
    
    # Add user message to context
    context.append(('human', user_message))
    
    # Retrieve relevant documents from RAG
    documents = retriever.invoke(user_message)
    doc_text = "\n".join(doc.page_content for doc in documents)
    doc_text = "Usa questo documento per rispondere e se non ci riesci rispondi boh: " + doc_text
    
    # Optionally include other characters' conversations
    if include_other_contexts:
        other_contexts = get_other_characters_context(character_name)
        other_contexts_text = format_other_contexts_for_prompt(other_contexts)
        doc_text += other_contexts_text
    
    # Update RAG context
    context[1] = ('system', doc_text)
    
    # Get AI response with retry logic for memory errors
    max_retries = 3
    response = None
    
    for attempt in range(max_retries):
        try:
            response = lcmodel.invoke(context)
            break  # Success!
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a memory error
            if "model requires more system memory" in error_str or "memory" in error_str:
                logging.warning(f"[{character_name}] ⚠️ Memory error on attempt {attempt + 1}/{max_retries}")
                
                if attempt < max_retries - 1:
                    # Try to free up memory by pruning context aggressively
                    original_length = len(context)
                    while len(context) > 4:  # Keep only system prompts + last exchange
                        context.pop(2)
                        if len(context) > 3:
                            context.pop(2)
                    
                    logging.info(f"[{character_name}] Pruned context from {original_length} to {len(context)} messages")
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    logging.error(f"[{character_name}] ❌ Failed after {max_retries} attempts")
                    # Remove the user message we just added since we failed
                    if context and context[-1][0] == 'human':
                        context.pop()
                    return "Mi dispiace, il modello è sovraccarico al momento. Prova a:\n1. Aspettare qualche secondo\n2. Fare domande più brevi\n3. Usare il comando: taskkill /F /IM ollama.exe && ollama serve"
            else:
                # Different error, don't retry
                logging.error(f"[{character_name}] Error: {e}")
                if context and context[-1][0] == 'human':
                    context.pop()
                return f"Errore nella generazione della risposta: {str(e)[:100]}"
    
    if response is None:
        if context and context[-1][0] == 'human':
            context.pop()
        return "Errore: nessuna risposta dal modello"
    
    # Add AI response to context
    context.append(('ai', response.content))
    
    # Context management - keep conversation manageable
    if len(context) > config['max_context_length']:
        # Remove oldest user-assistant pair (keep system prompts)
        context.pop(2)
        context.pop(2)
        logging.info(f"[{character_name}] Context pruned to maintain size")
    
    logging.info(f"[{character_name}] AI: {response.content[:100]}...")
    
    return response.content




@app.route('/')
def index():
    return render_template('multichar.html')

@app.route('/audio')
def index_Audio():
    return render_template('index.html')

@app.route('/chat/Alice', methods=["POST"])
def chat_alice():
    """Alice character endpoint"""
    data = request.get_json()
    user_question = data.get('question', 'No question')
    include_others = data.get('include_other_contexts', True)
    
    bot_response_text = asyncio.run(
        userRequest(user_question, 'Alice', check_safety=False, include_other_contexts=include_others)
    )
    
    return jsonify({
        "answer": bot_response_text,
        "character": "Alice"
    })



@app.route('/chat/Arthur', methods=["POST"])
def chat_arthur():
    """Arthur character endpoint"""
    data = request.get_json()
    user_question = data.get('question', 'No question')
    include_others = data.get('include_other_contexts', True)
    
    bot_response_text = asyncio.run(
        userRequest(user_question, 'Arthur', check_safety=False, include_other_contexts=include_others)
    )
    
    return jsonify({
        "answer": bot_response_text,
        "character": "Arthur"
    })


@app.route('/chat/Ruby', methods=["POST"])
def chat_ruby():
    """Ruby character endpoint"""
    data = request.get_json()
    user_question = data.get('question', 'No question')
    include_others = data.get('include_other_contexts', True)
    
    bot_response_text = asyncio.run(
        userRequest(user_question, 'Ruby', check_safety=False, include_other_contexts=include_others)
    )
    
    return jsonify({
        "answer": bot_response_text,
        "character": "Ruby"
    })




@app.route('/get_context/<character>', methods=["GET"])
def get_character_context(character):
    """Get the conversation history for a specific character"""
    if character not in dictContexts:
        return jsonify({"error": "Character not found"}), 404
    
    context = dictContexts[character]
    conversation = context[2:]
    
    return jsonify({
        "character": character,
        "conversation": [{"role": role, "content": content} for role, content in conversation]
    })


@app.route('/get_all_contexts', methods=["GET"])
def get_all_contexts():
    """Get all characters' conversation histories"""
    all_contexts = {}
    for char_name, context in dictContexts.items():
        conversation = context[2:]  # Skip system prompts
        all_contexts[char_name] = [
            {"role": role, "content": content} for role, content in conversation
        ]
    
    return jsonify(all_contexts)


@app.route('/clear_context/<character>', methods=["POST"])
def clear_character_context(character):
    """Clear conversation history for a specific character (keep system prompts)"""
    if character not in dictContexts:
        return jsonify({"error": "Character not found"}), 404
    
    context = dictContexts[character]
    dictContexts[character] = context[:2]
    
    return jsonify({
        "message": f"Context cleared for {character}",
        "character": character
    })


@app.route('/available_characters', methods=["GET"])
def available_characters():
    """Get list of available characters"""
    return jsonify({
        "characters": list(dictContexts.keys())
    })



@app.route('/chat/audio/<character>', methods=['POST'])
def chat_audio_character(character):
    """
    Audio endpoint for specific character
    Accepts raw audio data
    """
    if character not in dictContexts:
        return jsonify({"error": "Character not found"}), 404
    
    if not HAS_AUDIO:
        return jsonify({"error": "Audio features not available"}), 501
    
    audio = request.get_data()
    
    os.makedirs('./audio', exist_ok=True)
    with open('./audio/user_input.wav', 'wb') as f:
        f.write(audio)
    
    message = speech_to_text('./audio/user_input.wav')
    response_text = asyncio.run(userRequest(message, character, check_safety=False))
    audio_response = text_to_speech(response_text)
    
    return jsonify({
        "text": response_text,
        "audio": audio_response,
        "character": character,
        "emotion": "NEUTRO"
    })



@app.route('/send_message', methods=["POST", "GET"])
def receive_message():
    """Legacy endpoint - defaults to Alice"""
    data = request.get_json()
    user_question = data.get('question', 'No question')
    
    bot_response_text = asyncio.run(
        userRequest(user_question, 'Alice', check_safety=False)
    )
    
    return jsonify({
        "answer": bot_response_text 
    })


@app.route('/aicompanion', methods=['POST'])
def aicompanion():
    """Legacy generic endpoint - defaults to Alice"""
    data = request.get_json()
    message = data.get('message', '')
    audio_b64 = data.get('audio')
    
    # Handle audio input if provided
    if HAS_AUDIO and audio_b64 not in ["", None]:
        os.makedirs('./audio', exist_ok=True)
        
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with open('./audio/user_input.wav', 'wb') as f:
                f.write(audio_bytes)
            
            message = speech_to_text('./audio/user_input.wav')
            logging.info(f"Transcribed: {message}")
        except Exception as e:
            logging.error(f"Audio processing error: {e}")
            return jsonify({"error": "Audio processing failed"}), 400
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    # Process request with Alice (default character)
    response_text = asyncio.run(userRequest(message, 'Alice', check_safety=False))
    
    # Generate audio response if available
    audio_response = None
    if HAS_AUDIO:
        audio_response = text_to_speech(response_text)
    
    return jsonify({
        "response": response_text,
        "audio": audio_response,
        "emotion": "NEUTRO"
    })


if __name__ == '__main__':
    logging.info(f"Starting server on port {config['server_port']}")
    logging.info(f"Audio features: {'Enabled' if HAS_AUDIO else 'Disabled'}")
    logging.info(f"RAG database: {config['rag_db_path']}")
    logging.info(f"Available characters: {', '.join(dictContexts.keys())}")
    
    app.run(
        host="0.0.0.0",
        port=config['server_port'], 
        debug=True
    )
