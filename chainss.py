try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from decouple import config
    import os
except ImportError as e:
    print("Missing required packages. Please install them using:")
    print("pip install langchain-google-genai python-decouple langchain-core")
    exit(1)

# Load environment variables
try:
    os.environ["GOOGLE_API_KEY"] = config('GOOGLE_API_KEY')
except:
    print("Error: GOOGLE_API_KEY not found in .env file")
    print("Please make sure you have a .env file with GOOGLE_API_KEY=your_key_here")
    exit(1)

def create_chain():
    # Initialize the Google GenerativeAI model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,  # Reduced temperature for more factual responses
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
    except Exception as e:
        print(f"Error initializing Google AI model: {str(e)}")
        exit(1)

    # Create prompt template with additional context
    prompt = ChatPromptTemplate.from_template("""
Please provide accurate and up-to-date information for the following query. 
If the information involves recent political positions or appointments, 
please verify the timing and accuracy of your response.

Query: {input}
""")

    # Create the chain
    chain = prompt | llm

    return chain

def process_user_input(chain, user_input):
    """Process user input through the chain"""
    try:
        response = chain.invoke({"input": user_input})
        return response.content
    except Exception as e:
        return f"Error processing input: {str(e)}"

def main():
    # Create the chain
    chain = create_chain()
    
    print("LangChain Google AI Chat (type 'quit' to exit)")
    print("-" * 50)
    print("Note: For questions about current political positions,")
    print("please verify the information from official sources.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        response = process_user_input(chain, user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()