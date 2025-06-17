import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import re # Used for robust parsing of LLM output

# --- Setup ---
# Your Gemini API key should be securely added to Streamlit Cloud secrets.
# [secrets]
# GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
try:
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add 'GOOGLE_API_KEY' to your app's secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash") # Using flash for faster responses

st.set_page_config(
    page_title="Simple Feedback Analyzer",
    layout="wide",
    page_icon="✨"
)

# --- Helper Functions (Cache for Performance) ---
@st.cache_data(show_spinner=False)
def get_sentiment_and_score(text_series):
    """Applies sentiment analysis to a Pandas Series of texts."""
    sentiments, scores = [], []
    for text in text_series.dropna().astype(str):
        if not text.strip():
            sentiments.append("Neutral"); scores.append(0.0)
            continue
        prompt = f"Analyze sentiment as Positive, Negative, or Neutral, and score as 1, 0, or -1. Format: Sentiment: [Sentiment], Score: [Score]\nFeedback: \"{text}\""
        try:
            response = model.generate_content(prompt)
            result = response.text.strip()
            s_match = re.search(r"Sentiment:\s*([A-Za-z]+)", result)
            sc_match = re.search(r"Score:\s*([-+]?\d*\.?\d+)", result)
            sentiments.append(s_match.group(1).strip() if s_match else "Unknown")
            scores.append(float(sc_match.group(1).strip()) if sc_match else np.nan)
        except Exception:
            sentiments.append("Unknown"); scores.append(np.nan)
    return pd.Series(sentiments, index=text_series.index), pd.Series(scores, index=text_series.index)

@st.cache_data(show_spinner=False)
def summarize_feedback(feedback_text_list, creativity=0.4):
    """Summarizes a list of feedback texts into key themes."""
    joined_feedback = "\n".join(feedback_text_list).strip()
    if not joined_feedback:
        return "No sufficient feedback to summarize."
    prompt = f"Summarize the main themes from the following feedback as a bullet-point list:\n\n{joined_feedback}\n\nSummary:"
    try:
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=creativity))
        return response.text.strip()
    except Exception:
        return "Could not generate summary."

def plot_wordcloud(responses):
    """Generates and displays a word cloud."""
    text = " ".join(responses.dropna().astype(str).tolist())
    if not text.strip():
        st.info("Not enough text to generate a word cloud.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- Main App Logic ---
st.title("✨ Simple AI Feedback Analyzer")
st.markdown("Upload your feedback file (CSV/Excel). This app will analyze open-ended text fields for sentiment and provide summaries using AI.")

uploaded_file = st.file_uploader("📂 Upload Feedback File", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.subheader("📄 Raw Feedback Preview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Analysis Dashboard")

    text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].astype(str).apply(len).mean() > 20]

    if not text_columns:
        st.warning("No suitable open-ended text columns found for analysis. Please ensure your file has text columns with feedback.")
    else:
        selected_text_column = st.selectbox("Select Text Column for Analysis:", text_columns)

        if st.button("Run Analysis", help="Click to start sentiment analysis and summarization."):
            with st.spinner(f"Analyzing '{selected_text_column}'... This may take a moment for larger datasets."):
                # Get sentiments and scores for the selected column
                sentiments, scores = get_sentiment_and_score(df[selected_text_column])
                df[f'{selected_text_column}_Sentiment'] = sentiments
                df[f'{selected_text_column}_Score'] = scores

                st.markdown(f"---")
                st.subheader(f"Sentiment Analysis for '{selected_text_column}'")

                # Sentiment Distribution Chart
                sentiment_counts = df[f'{selected_text_column}_Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                fig_sentiment = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                                       title=f"Sentiment Distribution for '{selected_text_column}'",
                                       color_discrete_map={"Positive":"green", "Negative":"red", "Neutral":"blue", "Unknown":"gray"})
                st.plotly_chart(fig_sentiment, use_container_width=True)

                # Average Score
                avg_score = df[f'{selected_text_column}_Score'].mean()
                if not np.isnan(avg_score):
                    st.info(f"**Average Sentiment Score:** {avg_score:.2f} (closer to 1 is more positive, -1 more negative)")

                # Word Cloud
                st.markdown("#### Word Cloud")
                plot_wordcloud(df[selected_text_column])

                # Key Themes/Summary
                st.markdown("#### Key Themes/Summary (LLM-generated)")
                summary_responses = df[selected_text_column].dropna().tolist()
                # Limit responses sent to LLM for summary to max_rows_to_process for speed
                summary = summarize_feedback(summary_responses[:st.session_state.get('max_rows_to_process', 100)])
                st.success(summary)

                # Actionable Recommendations (from Negative feedback)
                st.markdown("#### Actionable Recommendations (from Negative Feedback)")
                negative_responses = df[df[f'{selected_text_column}_Sentiment'] == 'Negative'][selected_text_column].dropna().tolist()
                if negative_responses:
                    negative_summary = summarize_feedback(negative_responses[:st.session_state.get('max_rows_to_process', 100)], creativity=0.5)
                    recommendations_prompt = f"Based on this summary of negative feedback, provide 3-5 actionable recommendations:\n\n{negative_summary}\n\nRecommendations:"
                    try:
                        recommendations = model.generate_content(recommendations_prompt, generation_config=genai.types.GenerationConfig(temperature=0.6)).text.strip()
                        st.warning(recommendations)
                    except Exception:
                        st.info("Could not generate recommendations based on negative feedback.")
                else:
                    st.info("No negative feedback found to generate recommendations.")

else:
    st.info("Upload a CSV or Excel file to begin analyzing your feedback!")
    # Optional: Display a small sample if no file is uploaded, for a better first impression
    st.markdown("---")
    st.markdown("No file? Here's how sample data looks:")
    sample_data = pd.DataFrame({
        "Feedback": [
            "The service was excellent, very quick response!",
            "Had to wait a long time, very frustrating experience.",
            "Features are great, but the UI is confusing.",
            "Super friendly staff, resolved my issue immediately.",
            "Product is buggy, needs a lot of fixes."
        ]
    })
    st.dataframe(sample_data, use_container_width=True)
