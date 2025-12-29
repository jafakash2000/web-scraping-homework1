import streamlit as st

import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Web Scraping Dashboard")
st.title("🛒 Web Scraping Dashboard with Hugging Face Sentiment Analysis")
st.markdown("**Your Data Mining Homework - COMPLETE WITH AI!**")

# Load data
products = pd.read_csv("products.csv")
reviews = pd.read_csv("reviews.csv")
testimonials = pd.read_csv("testimonials.csv")

# Sidebar metrics
st.sidebar.title("📊 Summary")
st.sidebar.metric("Products", len(products))
st.sidebar.metric("Reviews", len(reviews))
st.sidebar.metric("Testimonials", len(testimonials))

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Section:", ["Products", "Reviews", "Testimonials"])

# Load Hugging Face sentiment model (cached for speed)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", 
                   model="distilbert-base-uncased-finetuned-sst-2-english",
                   device=-1)

sentiment_pipeline = load_sentiment_model()

if page == "Products":
    st.header("📦 Products (25 items)")
    st.dataframe(products, use_container_width=True)

elif page == "Reviews":
    st.header("⭐ Reviews - Advanced Sentiment Analysis")
    
    # Check if year and month columns exist
    if 'month' in reviews.columns and 'year' in reviews.columns:
        reviews_2023 = reviews[reviews['year'] == 2023]
        
        # Month slider
        month = st.slider("📅 Filter by Month (2023):", 1, 12, 5)
        filtered_reviews = reviews_2023[reviews_2023['month'] == month]
        
        st.write(f"**Showing {len(filtered_reviews)} reviews from month {month}**")
        
        if len(filtered_reviews) > 0:
            # Sentiment Analysis
            st.subheader("🤖 Hugging Face Transformer Sentiment Analysis")
            st.write("Model: `distilbert-base-uncased-finetuned-sst-2-english`")
            
            sentiments = []
            scores = []
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for idx, review_text in enumerate(filtered_reviews['text'].values):
                try:
                    result = sentiment_pipeline(review_text[:512])  # Limit to 512 chars
                    label = result[0]['label']
                    score = result[0]['score']
                    sentiments.append(label)
                    scores.append(score)
                except Exception as e:
                    sentiments.append("NEUTRAL")
                    scores.append(0.5)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(filtered_reviews))
            
            # Add results to dataframe
            filtered_reviews = filtered_reviews.copy()
            filtered_reviews['Sentiment'] = sentiments
            filtered_reviews['Confidence'] = scores
            
            # Display detailed table
            st.dataframe(filtered_reviews[['date_str', 'text', 'Sentiment', 'Confidence']], 
                        use_container_width=True)
            
            # Calculate metrics
            positive_count = (pd.Series(sentiments) == 'POSITIVE').sum()
            negative_count = (pd.Series(sentiments) == 'NEGATIVE').sum()
            avg_confidence = pd.Series(scores).mean()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Positive Reviews", positive_count)
            with col2:
                st.metric("❌ Negative Reviews", negative_count)
            with col3:
                st.metric("🎯 Avg Confidence Score", f"{avg_confidence:.3f}")
            
            # Bar Chart with Confidence Score
            st.subheader("📊 Sentiment Distribution with Confidence Scores")
            
            fig = go.Figure()
            
            positive_conf = pd.Series(scores)[pd.Series(sentiments) == 'POSITIVE'].mean()
            negative_conf = pd.Series(scores)[pd.Series(sentiments) == 'NEGATIVE'].mean()
            
            fig.add_trace(go.Bar(
                x=['Positive', 'Negative'],
                y=[positive_count, negative_count],
                text=[f"Count: {positive_count}<br>Confidence: {positive_conf:.3f}" if positive_count > 0 else "0",
                      f"Count: {negative_count}<br>Confidence: {negative_conf:.3f}" if negative_count > 0 else "0"],
                textposition='auto',
                marker=dict(
                    color=['green', 'red'],
                    opacity=0.8
                )
            ))
            
            fig.update_layout(
                title=f"Sentiment Analysis Results - Month {month} 2023",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Sentiment analysis complete! {positive_count} positive, {negative_count} negative reviews.")
        else:
            st.warning("No reviews found for this month.")
    else:
        st.error("Missing month/year columns in reviews data.")

elif page == "Testimonials":
    st.header("💬 Testimonials (10)")
    for i, row in testimonials.iterrows():
        st.write(f"**{i+1}.** {row['text']}")
    st.success("✅ ASSIGNMENT COMPLETE!")

# Footer
st.markdown("---")
st.markdown("""
**Technologies Used:**
- 🐍 Python, Streamlit
- 🤖 Hugging Face Transformers (DistilBERT)
- 📊 Plotly for visualizations
- 🔗 Web Scraping with Selenium + BeautifulSoup
""")
