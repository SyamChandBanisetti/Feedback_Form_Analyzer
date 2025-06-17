import os
import io
import streamlit as st
import pandas as pd
#from dotenv import load_dotenv
import google.generativeai as genai
from streamlit_chat import message
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_feedback import streamlit_feedback
import numpy as np

# -------------------- Setup --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(
    page_title="Executive Feedback Analyzer",
    layout="wide",
    page_icon="👑"
)

# -------------------- Helper Functions --------------------
# @st.cache_data is crucial here for performance
@st.cache_data(show_spinner=False)
def get_sentiment_and_score(text):
    """
    Classifies sentiment and assigns a numerical score.
    Score: -1 (Negative), 0 (Neutral), 1 (Positive)
    """
    if not isinstance(text, str) or not text.strip():
        return "Neutral", 0.0 # Handle empty or non-string inputs gracefully

    prompt = f"""Analyze the sentiment of the following feedback and classify it as Positive, Negative, or Neutral.
    Then, provide a sentiment score: -1 for Negative, 0 for Neutral, and 1 for Positive.
    Return the output in the exact format: Sentiment: [Sentiment], Score: [Score]
    Feedback: "{text}"\n"""
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        sentiment = "Unknown"
        score = np.nan

        # Robust parsing of the LLM output
        sentiment_match = re.search(r"Sentiment:\s*([A-Za-z]+)", result)
        score_match = re.search(r"Score:\s*([-+]?\d*\.?\d+)", result)

        if sentiment_match:
            sentiment = sentiment_match.group(1).strip()
        if score_match:
            try:
                score = float(score_match.group(1).strip())
            except ValueError:
                pass # Score remains NaN if conversion fails

        return sentiment, score
    except Exception as e:
        # st.warning(f"Error classifying sentiment for '{text[:50]}...': {e}") # Too many warnings can clutter
        return "Unknown", np.nan

@st.cache_data(show_spinner=False)
def summarize_question(question, responses, n_points=5, creativity=0.7):
    joined = "\n".join(responses.dropna().astype(str).tolist())
    if not joined.strip():
        return "No sufficient responses to summarize."
    prompt = f"""You are analyzing responses to this question: "{question}".
    Summarize the top {n_points} recurring themes or takeaways as a bullet-point list:

    {joined}

    Summary:"""
    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return result.text.strip()
    except Exception as e:
        # st.warning(f"Error summarizing question '{question}': {e}")
        return "Could not extract insights."

@st.cache_data(show_spinner=False)
def extract_topics(responses, n_topics=3, creativity=0.7):
    joined = "\n".join(responses.dropna().astype(str).tolist())
    if not joined.strip():
        return []
    prompt = f"""Analyze the following feedback responses and identify up to {n_topics} main recurring topics or categories.
    List them as a comma-separated list of keywords or short phrases.
    Responses:
    {joined}

    Topics:"""
    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return [topic.strip() for topic in result.text.strip().split(',') if topic.strip()]
    except Exception as e:
        # st.warning(f"Error extracting topics: {e}")
        return []

@st.cache_data(show_spinner=False)
def extract_key_phrases(responses, n_phrases=5, creativity=0.7):
    joined = "\n".join(responses.dropna().astype(str).tolist())
    if not joined.strip():
        return []
    prompt = f"""From the following feedback responses, identify the top {n_phrases} most frequently mentioned and significant key phrases or keywords. These should be direct quotes or very close paraphrases of recurring ideas.
    List them as a bullet-point list.
    Responses:
    {joined}

    Key Phrases:"""
    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        # Clean up bullet points from LLM output
        phrases = [phrase.strip().lstrip('- ').replace('* ', '') for phrase in result.text.strip().split('\n') if phrase.strip()]
        return [phrase for phrase in phrases if phrase] # Remove empty strings
    except Exception as e:
        # st.warning(f"Error extracting key phrases: {e}")
        return []

@st.cache_data(show_spinner=False)
def generate_recommendations(negative_feedback_summary, creativity=0.7):
    if not negative_feedback_summary.strip() or "No sufficient responses" in negative_feedback_summary:
        return "No specific negative feedback identified for recommendations."
    prompt = f"""Based on the following summary of negative feedback, provide 3-5 actionable recommendations for improvement.
    Summary of negative feedback:
    {negative_feedback_summary}

    Actionable Recommendations:"""
    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return result.text.strip()
    except Exception as e:
        # st.warning(f"Error generating recommendations: {e}")
        return "Could not generate recommendations."


def load_feedback_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def plot_wordcloud(responses):
    text = " ".join(responses.dropna().astype(str).tolist())
    if not text.strip():
        st.info("Not enough text to generate a word cloud.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def download_analysis(df):
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0) # Important: rewind the buffer
    st.download_button(
        "⬇️ Download Analysis as Excel",
        output.getvalue(),
        file_name="feedback_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_sample_data():
    sample = pd.DataFrame({
        "Timestamp": ["2024-01-15", "2024-01-16", "2024-02-01", "2024-02-10", "2024-03-05", "2024-03-10", "2024-04-01", "2024-04-15", "2024-05-01", "2024-05-10"],
        "Service Quality": [
            "Excellent service, very prompt and helpful!",
            "Good, but wait times were a bit long. Staff seemed overwhelmed.",
            "Friendly staff, no complaints, resolved issue quickly.",
            "Service was poor, really slow response. Unacceptable delays.",
            "Very efficient and helpful, problem solved in minutes.",
            "Okay service, but the online portal is confusing.",
            "Quick and professional, very satisfied with the assistance.",
            "Long hold times, very frustrating experience.",
            "Outstanding support! Knowledgeable and polite.",
            "Customer service needs improvement, couldn't get a clear answer."
        ],
        "Product Features": [
            "Love the new features, exactly what I needed. Very intuitive.",
            "Features are okay, could be more intuitive. Some bugs noticed.",
            "Works as expected, no issues. Simple and reliable.",
            "Missing key features I use daily. Very basic compared to competitors.",
            "Great new update! Performance is much better.",
            "Some features are clunky, need better integration.",
            "The search function is terrible, hard to find what I need.",
            "Very comprehensive set of features, impressed by the depth.",
            "User interface is not friendly, steep learning curve.",
            "Could use more customization options."
        ],
        "Overall Experience (1-5)": [5, 4, 5, 2, 5, 3, 5, 2, 4, 3]
    })
    st.write("**Sample Data Preview:**")
    st.dataframe(sample, use_container_width=True)
    buffer = io.StringIO()
    sample.to_csv(buffer, index=False)
    st.download_button("Download Sample CSV", buffer.getvalue(), file_name="sample_feedback_exec.csv")

import re # Import regex for robust parsing

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/feedback.png", width=80)
    st.title("👑 Executive Feedback Analyzer")
    st.markdown("Upload your feedback file (CSV/Excel). This tool analyzes feedback for executive-level insights, including sentiment, key themes, and actionable recommendations.")
    uploaded_file = st.file_uploader("📂 Upload Feedback File", type=["csv", "xlsx"], help="Accepted formats: CSV, XLSX")
    st.markdown("---")
    st.markdown("Don't have a file? Download a sample below to see how it works.")
    show_sample_data()
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    llm_creativity = st.slider(
        "LLM Creativity (Temperature)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.1,
        help="Higher values make summaries/topics/recommendations more creative and less deterministic. Lower values are more focused and factual. Recommended for executive insights: 0.3-0.5"
    )
    max_rows_to_process = st.number_input(
        "Max Rows for LLM Analysis (Open-ended)",
        min_value=10,
        max_value=1000, # Increased max for larger datasets, can be even higher
        value=100, # Default to 100 for quicker analysis
        step=10,
        help="To speed up analysis, limit the number of open-ended responses processed by the LLM. Set to a higher value for full dataset analysis (may take longer)."
    )
    st.markdown("---")
    st.markdown("### 💬 Ask FeedbackBot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_prompt = st.text_input("Ask something about feedback analysis", key="chat_input", help="E.g., 'What is the most effective way to address negative feedback?' or 'How often should we analyze feedback?'")
    if chat_prompt:
        st.session_state.chat_history.append(("user", chat_prompt))
        with st.spinner("FeedbackBot is thinking..."):
            try:
                reply = model.generate_content(chat_prompt).text.strip()
            except Exception as e:
                reply = f"Sorry, I encountered an error. Please try again. ({e})"
        st.session_state.chat_history.append(("bot", reply))
    if st.session_state.chat_history:
        # Display chat history in reverse order to show latest message at the bottom
        for sender, msg in st.session_state.chat_history[::-1]:
            message(msg, is_user=(sender == "user"), key=f"{sender}_{len(msg)}_{hash(msg[:50])}") # Added hash for more robust key

# -------------------- Main App --------------------
st.title("👑 Gemini Executive Feedback Analyzer")

if uploaded_file:
    df = load_feedback_file(uploaded_file)
    original_columns = df.columns.tolist() # Store original columns
    st.subheader("📄 Raw Feedback Preview")
    with st.expander("Show/hide raw data"):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Executive Summary Dashboard")

    analysis_df = df.copy()
    sentiment_columns = [] # To store names of columns with sentiment scores
    all_positive_responses = pd.Series(dtype=str)
    all_negative_responses = pd.Series(dtype=str)

    # Ask for date column for trend analysis
    date_column_options = ["None (Skip Trend Analysis)"] + original_columns
    selected_date_column = st.selectbox(
        "Select a **Date Column** for Trend Analysis (if applicable):",
        date_column_options,
        help="If your feedback has a date, select the column to see sentiment trends over time. The column should be in a recognizable date format."
    )

    if selected_date_column != "None (Skip Trend Analysis)":
        try:
            # Attempt to convert to datetime, inferring format
            analysis_df[selected_date_column] = pd.to_datetime(analysis_df[selected_date_column], errors='coerce')
            # Drop rows where date parsing failed
            initial_rows = len(analysis_df)
            analysis_df.dropna(subset=[selected_date_column], inplace=True)
            dropped_rows = initial_rows - len(analysis_df)
            if dropped_rows > 0:
                st.warning(f"Date column '{selected_date_column}' parsed with {dropped_rows} invalid date entries removed.")
            else:
                st.success(f"Date column '{selected_date_column}' successfully parsed.")
            analysis_df.sort_values(by=selected_date_column, inplace=True)
        except Exception as e:
            st.error(f"Could not parse date column '{selected_date_column}'. Please ensure it's in a valid date format. Error: {e}")
            selected_date_column = "None (Skip Trend Analysis)" # Revert if parsing fails

    # --- Perform detailed question-wise analysis ---
    st.markdown("---")
    st.subheader("🔍 Detailed Question-wise Analysis")

    for column in original_columns:
        # Skip the date column from detailed analysis if it was selected
        if column == selected_date_column and selected_date_column != "None (Skip Trend Analysis)":
            continue

        st.markdown(f"---")
        st.markdown(f"### ❓ Question: `{column}`")
        responses = analysis_df[column].dropna()

        if responses.empty:
            st.info("No responses for this question.")
            continue

        # Try to infer if it's numerical (e.g., ratings)
        is_numerical = pd.api.types.is_numeric_dtype(responses) and responses.nunique() > 1 and responses.max() > 1

        # Check for open-ended vs. short answer/MCQ based on unique values and average length
        avg_len = responses.astype(str).str.len().mean()
        # Heuristic: more than 20 chars avg AND many unique values OR very high unique ratio
        is_open_ended = (avg_len > 20 and responses.nunique() > 5) or (responses.nunique() / len(responses) > 0.5)

        if is_numerical and not is_open_ended:
            st.markdown("**📊 Numerical Responses Detected (e.g., Ratings)**")
            counts = responses.value_counts().sort_index().reset_index()
            counts.columns = [column, "Count"]
            fig = px.bar(counts, x=column, y="Count", color=column, title=f"Distribution of '{column}' Responses")
            st.plotly_chart(fig, use_container_width=True)

            if len(responses) > 0:
                st.info(f"**Average {column}:** {responses.mean():.2f}")
                st.info(f"**Median {column}:** {responses.median():.2f}")

        elif is_open_ended:
            st.markdown("**🗣 Open-ended Responses Detected**")

            # --- Apply Max Rows to Process ---
            # Sample responses if there are too many, for performance. Keep original index for updates.
            responses_to_process = responses.sample(n=min(len(responses), max_rows_to_process), random_state=42)
            st.info(f"Analyzing {len(responses_to_process)} of {len(responses)} open-ended responses for this question to speed up analysis. Adjust 'Max Rows for LLM Analysis' in sidebar for more.")

            current_sentiments = []
            current_scores = []
            progress_text = f"Analyzing sentiments for '{column}'..."
            my_bar = st.progress(0, text=progress_text)

            for i, text in enumerate(responses_to_process):
                sentiment, score = get_sentiment_and_score(text)
                current_sentiments.append(sentiment)
                current_scores.append(score)
                my_bar.progress((i + 1) / len(responses_to_process), text=f"{progress_text} ({i+1}/{len(responses_to_process)})")
            my_bar.empty() # Clear the progress bar after completion

            # Create temporary Series aligned with original 'responses' index, then update analysis_df
            sentiment_series = pd.Series(current_sentiments, index=responses_to_process.index)
            score_series = pd.Series(current_scores, index=responses_to_process.index)

            analysis_df.loc[responses_to_process.index, column + "_Sentiment"] = sentiment_series
            analysis_df.loc[responses_to_process.index, column + "_Sentiment_Score"] = score_series

            sentiment_columns.append(column + "_Sentiment_Score")

            # Aggregate all positive/negative responses for executive summary (using all responses, not just processed subset)
            # Filter the *original* responses based on sentiments from *processed* subset
            all_positive_responses = pd.concat([all_positive_responses, responses_to_process[sentiment_series == 'Positive'].reset_index(drop=True)])
            all_negative_responses = pd.concat([all_negative_responses, responses_to_process[sentiment_series == 'Negative'].reset_index(drop=True)])


            # Display charts/summaries based on the processed subset
            sentiment_counts = pd.Series(current_sentiments).value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment", title=f"Sentiment Distribution for '{column}' (Sampled)")
            st.plotly_chart(fig, use_container_width=True)

            if len(current_scores) > 0 and not pd.Series(current_scores).isnull().all():
                avg_score = np.nanmean(current_scores)
                st.info(f"**Average Sentiment Score for '{column}' (Sampled):** {avg_score:.2f} (closer to 1 is more positive, -1 more negative)")

            with st.expander(f"Show Word Cloud for '{column}' (Sampled)"):
                plot_wordcloud(responses_to_process) # Use sampled responses for word cloud

            st.markdown("**💡 Key Takeaways (LLM Summary - Sampled)**")
            n_summary_points = st.slider(f"Number of summary points for '{column}'", 3, 10, 5, key=f"summary_slider_{column}")
            summary = summarize_question(column, responses_to_process, n_points=n_summary_points, creativity=llm_creativity)
            st.success(summary)

            st.markdown("**🎯 Top Key Phrases (LLM Extracted - Sampled)**")
            n_phrases_extract = st.slider(f"Number of key phrases for '{column}'", 3, 10, 5, key=f"phrases_slider_{column}")
            key_phrases = extract_key_phrases(responses_to_process, n_phrases=n_phrases_extract, creativity=llm_creativity)
            if key_phrases:
                for phrase in key_phrases:
                    st.markdown(f"- {phrase}")
            else:
                st.info("Could not extract distinct key phrases from sampled responses.")

            # Recommendations only for questions with negative sentiment
            negative_responses_for_column = responses_to_process[sentiment_series == 'Negative']
            if not negative_responses_for_column.empty:
                st.markdown("**🛠️ Actionable Recommendations (from Negative Feedback - Sampled)**")
                negative_summary_for_column = summarize_question(column, negative_responses_for_column, n_points=3, creativity=llm_creativity)
                recommendations = generate_recommendations(negative_summary_for_column, creativity=llm_creativity)
                st.warning(recommendations)
            else:
                st.info("No negative feedback found in sampled responses for this question to generate recommendations.")

            with st.expander(f"🧾 Sample Responses for '{column}'"):
                # Still show original, random samples for display
                for i, resp in enumerate(responses.sample(min(5, len(responses)), random_state=42), 1):
                    st.markdown(f"- {resp}")

        else: # Default for other types, treat as categorical
            st.markdown("**📋 Categorical/Short Answer Detected**")
            counts = responses.value_counts().reset_index()
            counts.columns = [column, "Count"]
            fig = px.bar(counts, x=column, y="Count", color=column, title=f"Distribution of '{column}' Responses")
            st.plotly_chart(fig, use_container_width=True)

    # --- Executive Summary Dashboard Content ---
    st.markdown("---")
    st.subheader("📊 Executive Summary Dashboard - Aggregated Insights")
    st.info("Insights below are aggregated from the *sampled* open-ended responses.")

    if not all_positive_responses.empty or not all_negative_responses.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Top Success Factors")
            if not all_positive_responses.empty:
                st.info("Aggregated key phrases from all positive feedback:")
                success_factors = extract_key_phrases(all_positive_responses, n_phrases=5, creativity=llm_creativity)
                if success_factors:
                    for factor in success_factors:
                        st.markdown(f"- **{factor}**")
                else:
                    st.info("No significant success factors identified from sampled responses.")
            else:
                st.info("No positive feedback found across all open-ended questions in sampled data.")

        with col2:
            st.markdown("#### 🚩 Top Pain Points")
            if not all_negative_responses.empty:
                st.warning("Aggregated key phrases from all negative feedback:")
                pain_points = extract_key_phrases(all_negative_responses, n_phrases=5, creativity=llm_creativity)
                if pain_points:
                    for point in pain_points:
                        st.markdown(f"- **{point}**")
                else:
                    st.info("No significant pain points identified from sampled responses.")
            else:
                st.info("No negative feedback found across all open-ended questions in sampled data.")

        if not all_negative_responses.empty:
            st.markdown("#### 💡 Overall Actionable Recommendations")
            overall_negative_summary = summarize_question("Overall Negative Feedback", all_negative_responses, n_points=5, creativity=llm_creativity)
            overall_recommendations = generate_recommendations(overall_negative_summary, creativity=llm_creativity)
            st.success(overall_recommendations)
        else:
            st.info("No overall negative feedback from sampled data to generate recommendations.")
    else:
        st.info("No open-ended feedback found for executive summary (Success Factors/Pain Points) in sampled data.")


    # Cross-question correlation (numerical ratings vs. sentiment scores)
    numerical_rating_columns = [col for col in original_columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1 and df[col].max() > 1]

    if sentiment_columns and numerical_rating_columns:
        st.markdown("---")
        st.markdown("### 🤝 Correlation between Ratings and Sentiment")
        st.info("This section shows if higher ratings correlate with more positive sentiment in open-ended feedback (based on sampled data).")
        correlation_found = False
        for sent_col in sentiment_columns:
            original_q = sent_col.replace("_Sentiment_Score", "")
            for num_col in numerical_rating_columns:
                # Use analysis_df which now contains sentiment scores for the *sampled* rows
                temp_df = analysis_df[[num_col, sent_col]].dropna()
                if not temp_df.empty and len(temp_df) > 1: # Need at least 2 data points for correlation
                    correlation = temp_df[num_col].corr(temp_df[sent_col])
                    if not pd.isna(correlation):
                        st.write(f"- **Correlation between `{num_col}` and sentiment for `{original_q}`:** `{correlation:.2f}`")
                        st.caption("A positive correlation indicates higher ratings are associated with more positive sentiment.")
                        correlation_found = True
        if not correlation_found:
            st.info("No meaningful correlation data found between numerical ratings and open-ended sentiment.")


    # Trend Analysis over Time
    if selected_date_column != "None (Skip Trend Analysis)" and sentiment_columns:
        st.markdown("---")
        st.markdown("### 🗓 Sentiment Trends Over Time")
        st.info("Trend analysis based on *sampled* data.")
        for sent_col in sentiment_columns:
            original_q = sent_col.replace("_Sentiment_Score", "")
            # Ensure we're using analysis_df with potentially filtered/re-indexed rows
            trend_df = analysis_df.dropna(subset=[selected_date_column, sent_col]).copy()

            if not trend_df.empty:
                # Group by date and calculate mean sentiment score
                daily_sentiment = trend_df.groupby(selected_date_column)[sent_col].mean().reset_index()

                fig_trend = px.line(
                    daily_sentiment,
                    x=selected_date_column,
                    y=sent_col,
                    title=f"Average Sentiment Score for '{original_q}' Over Time (Sampled)"
                )
                fig_trend.update_layout(yaxis_title="Average Sentiment Score (-1 to 1)")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info(f"No sentiment data for '{original_q}' to plot trends over time with the selected date column from sampled data.")
    elif selected_date_column != "None (Skip Trend Analysis)" and not sentiment_columns:
        st.info("No open-ended questions with sentiment analysis found to plot trends.")

    st.markdown("---")
    st.subheader("💾 Download Analysis")
    download_analysis(analysis_df)

    # User feedback widget
    st.markdown("---")
    st.markdown("### 🙏 Rate this Analyzer")
    feedback = streamlit_feedback(feedback_type="thumbs", optional_text_label="Any suggestions?")
    if feedback:
        st.success("Thanks for your feedback!")

else:
    st.info("Please upload a feedback CSV or Excel file to begin.")
