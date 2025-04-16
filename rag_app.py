import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Career RAG Assistant 🎯", page_icon="💼", layout="centered")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def preprocess_career_data(df):
    df = df.fillna("")
    df['combined'] = df.apply(lambda row: ' '.join([str(val).lower() for val in row.values]), axis=1)
    return df

def create_vectorizer(df):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df['combined'])
    return vectorizer, vectors

def find_best_match(user_query, vectorizer, vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, vectors).flatten()
    index = similarities.argmax()
    score = similarities[index]
    return (df.iloc[index], score) if score > 0.3 else (None, score)

def configure_generative_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def refine_response(model, user_query, matched_data):
    context = """
    You are a career guidance assistant. Using the data below, generate a clear and structured response.
    Highlight job roles, required skills, and salary expectations if applicable.
    """
    prompt = f"{context}\n\nUser Query: {user_query}\nMatched Data: {matched_data}\nResponse:"
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("Career RAG Assistant 🎯")
    st.write("Upload your career dataset and ask questions based on its content!")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty.")
            return

        df = preprocess_career_data(df)
        vectorizer, vectors = create_vectorizer(df)

        API_KEY = st.secrets.get("GOOGLE_API_KEY")
        if not API_KEY:
            st.error("API key not found. Set it in Streamlit Secrets.")
            return

        model = configure_generative_model(API_KEY)

        st.markdown("### Conversation History")
        for msg in st.session_state.conversation:
            st.markdown(
                f"<div style='background-color: #e6f7ff; padding:10px; border-radius:10px; margin:5px 0;'>"
                f"<strong>{msg['role']}:</strong> {msg['content']}</div>", 
                unsafe_allow_html=True
            )

        user_query = st.text_input("Ask your question:", key="user_input")

        if user_query:
            st.session_state.conversation.append({"role": "User", "content": user_query})
            best_match, score = find_best_match(user_query, vectorizer, vectors, df)

            if best_match is not None:
                with st.spinner("Generating refined response..."):
                    refined = refine_response(model, user_query, best_match.to_dict())
                    st.session_state.conversation.append({"role": "Bot", "content": refined})
                    st.markdown(
                        f"<div style='background-color: #f0f0f0; padding:10px; border-radius:10px; margin:5px 0;'>"
                        f"<strong>Bot:</strong> {refined}</div>", 
                        unsafe_allow_html=True
                    )
            else:
                try:
                    fallback_prompt = f"User: {user_query}\nProvide career advice based on this query."
                    response = model.generate_content(fallback_prompt)
                    st.session_state.conversation.append({"role": "Bot", "content": response.text})
                    st.markdown(
                        f"<div style='background-color: #f0f0f0; padding:10px; border-radius:10px; margin:5px 0;'>"
                        f"<strong>Bot:</strong> {response.text}</div>", 
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
