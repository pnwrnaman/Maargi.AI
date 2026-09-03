import os 
from dotenv import load_dotenv 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder 
from langchain_core.messages import HumanMessage , SystemMessage , AIMessage 
load_dotenv()
#Embedding model 
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': False} 
)
#llama model
model_repo = os.getenv("LLAMA_MODEL_REPO", "meta-llama/Llama-3.1-8B-Instruct")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing in .env file!")

llm = HuggingFaceEndpoint(
    repo_id = model_repo,
    task= "Text Generation",
    max_new_tokens = 1024,
    temperature=0.7,
    do_sample=True,
    huggingfacehub_api_token=hf_token

)
maargi_llm= ChatHuggingFace( llm = llm)

#prompt template with convo memory 
system_prompt = """
You are MaargiAI, an expert, insider travel guide for Uttarakhand, India.
Your mission is to help users discover hidden gems, offbeat homestays, and unique trails.
You specialize in routes less traveled, such as the trek from Srinagar to Rudranath via Sagar village and Panar Bugyal.

Tone: Enthusiastic, highly knowledgeable, practical, and authentic.
Format: Use Markdown formatting. Use bullet points for itineraries, and highlight important safety or permit information.

CRITICAL RULE: Never invent places, trails, or facts. If a user asks about an area you don't have deep local knowledge of, state that clearly while offering the best general guidance possible.
"""
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ("human","{user_input}" )
])

# chain
maargi_chain = prompt | maargi_llm
 #helper fx


def generate_itinerary(query: str, chat_history: list = None) -> str:
    """
    Takes a user query and optional chat history,
    invokes the Llama chain, and returns the response string.
    """
    if chat_history is None:
        chat_history = []
        
    try:
        response = maargi_chain.invoke({
            "chat_history": chat_history,
            "user_input": query
        })
        return response.content
    except Exception as e:
        return f"Maargi AI encountered an error: {str(e)}"


# RUN LOCAL TEST

if __name__ == "__main__":
    import sys
    print("⏳ Step 1: Loading Local Embeddings...", flush=True)
    test_embedding = embeddings.embed_query("Sagar Village to Panar Bugyal")
    print(f"✅ Step 1 Complete! Vector size: {len(test_embedding)}", flush=True)
    
    print(f"\n⏳ Step 2: Connecting to Hugging Face model ({model_repo})...", flush=True)
    
    mock_history = [
        HumanMessage(content="I want to do a high-altitude trek from Srinagar to Rudranath."),
        AIMessage(content="That's a fantastic, challenging route! I highly recommend taking the route up through Sagar village and resting at Panar Bugyal before pushing for Rudranath. When are you planning to go?")
    ]
    
    test_query = "What gear should I rent in Srinagar before starting that specific route?"
    
    print("⏳ Step 3: Sending request to Llama (this may take 10-20s on first request)...", flush=True)
    result = generate_itinerary(test_query, chat_history=mock_history)
    print(f"\n🤖 Maargi AI Response:\n{result}", flush=True)