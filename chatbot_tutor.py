import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Google Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def show_tutor_chatbot():
    """Main interface for the AI-powered virtual tutor."""
    st.title("👨‍🏫 Virtual Tutor")
    
    st.info("I am an AI-powered tutor designed to help you with your studies. Feel free to ask me anything about your subjects or learning materials.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the Gemini Model
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = genai.GenerativeModel('models/gemini-2.5-flash')
        # Define the system prompt for the virtual tutor's persona
        st.session_state.persona = """You are a helpful and patient virtual tutor. Your purpose is to assist students with their learning.
        When a student asks a question, you should:
        - Provide a clear and concise explanation.
        - Use simple language and analogies to make complex topics easy to understand.
        - Avoid giving direct answers to homework or quiz questions. Instead, provide hints or guide the student to the solution.
        - Be encouraging and positive.
        - If a question is outside of your expertise (e.g., asking for personal opinions or real-time information), politely decline and redirect the student to a learning-related topic."""
        # Start a chat session with the defined persona
        st.session_state.chat = st.session_state.chat_model.start_chat(history=[])
        st.session_state.chat.send_message(st.session_state.persona)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask me anything about your lessons..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("tutor"):
            # Get response from the LLM
            with st.spinner("Thinking..."):
                response = st.session_state.chat.send_message(prompt, stream=True)
                response_text = ""
                for chunk in response:
                    response_text += chunk.text
                    st.markdown(response_text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "tutor", "content": response_text})

if __name__ == '__main__':
    show_tutor_chatbot()