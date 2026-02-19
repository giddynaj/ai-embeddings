from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key=os.getenv('OPENAI_API_KEY')

client=OpenAI(api_key=openai_api_key)

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

vector = embed("The dog chased the ball")
print(f"Vector dimensions: {len(vector)}")
