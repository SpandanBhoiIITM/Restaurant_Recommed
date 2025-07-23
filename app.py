import streamlit as st
import pandas as pd
import numpy as np
from utils.recommendation_engine import RestaurantRecommendationEngine
import os

# Configure page
st.set_page_config(
    page_title="🌟 Michelin Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for futuristic styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .restaurant-card {
        background: linear-gradient(135deg, #1e2139, #2a2d47);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    .restaurant-name {
        color: #00d4ff;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .restaurant-details {
        color: #ccc;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .similarity-score {
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .stat-box {
        text-align: center;
        background: rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #00d4ff;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    
    .stat-label {
        color: #ccc;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_recommendation_engine():
    """Load and initialize the recommendation engine with caching"""
    try:
        engine = RestaurantRecommendationEngine()
        return engine
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_restaurant_card(name, details, similarity_score=None):
    """Display a restaurant in a futuristic card format"""
    card_html = f"""
    <div class="restaurant-card">
        <div class="restaurant-name">🌟 {name.title()}</div>
        <div class="restaurant-details">
            <strong>Cuisine:</strong> {details.get('cuisine', 'Unknown').title()}<br>
            <strong>Location:</strong> {details.get('city', 'Unknown')}, {details.get('region', 'Unknown')}<br>
            <strong>Price Range:</strong> {details.get('price', 'Unknown')}<br>
            <strong>Year:</strong> {details.get('year', 'Unknown')}
        </div>
        {f'<div class="similarity-score">Similarity: {similarity_score:.2%}</div>' if similarity_score else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ MICHELIN RESTAURANT RECOMMENDER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your next culinary adventure with AI-powered recommendations</p>', unsafe_allow_html=True)
    
    # Load recommendation engine
    engine = load_recommendation_engine()
    
    if engine is None:
        st.error("Failed to load the recommendation engine. Please check the data files.")
        return
    
    # Display statistics
    stats = engine.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['total_restaurants']}</div>
            <div class="stat-label">Total Restaurants</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['unique_cuisines']}</div>
            <div class="stat-label">Cuisine Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['unique_cities']}</div>
            <div class="stat-label">Cities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['unique_countries']}</div>
            <div class="stat-label">Countries</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Search section
    st.markdown("### 🔍 Find Similar Restaurants")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Restaurant search with autocomplete suggestions
        all_restaurants = engine.get_all_restaurant_names()
        restaurant_input = st.selectbox(
            "Search for a restaurant or select from the list:",
            options=[""] + sorted(all_restaurants),
            format_func=lambda x: x.title() if x else "Type or select a restaurant...",
            key="restaurant_search"
        )
    
    with col2:
        num_recommendations = st.selectbox(
            "Number of recommendations:",
            options=[3, 5, 7, 10],
            index=1
        )
    
    # Manual text input as fallback
    if not restaurant_input:
        restaurant_input = st.text_input(
            "Or type a restaurant name:",
            placeholder="e.g., Le Bernardin, Noma, etc.",
            key="manual_input"
        )
    
    if restaurant_input:
        with st.spinner("🤖 Analyzing cuisine profiles and finding recommendations..."):
            recommendations = engine.recommend_restaurant(restaurant_input, top_n=num_recommendations)
            
            if isinstance(recommendations, str):
                # Error case
                st.error(recommendations)
                
                # Show suggestions for similar names
                suggestions = engine.get_restaurant_suggestions(restaurant_input)
                if suggestions:
                    st.info("Did you mean one of these restaurants?")
                    for suggestion in suggestions[:5]:
                        if st.button(f"🍽️ {suggestion.title()}", key=f"suggestion_{suggestion}"):
                            st.session_state.restaurant_search = suggestion
                            st.rerun()
            else:
                # Success case
                selected_restaurant = engine.find_closest_match(restaurant_input)
                restaurant_details = engine.get_restaurant_details(selected_restaurant)
                
                st.success(f"✨ Showing recommendations based on: **{selected_restaurant.title()}**")
                
                # Display the selected restaurant
                st.markdown("#### 🎯 Selected Restaurant")
                display_restaurant_card(selected_restaurant, restaurant_details)
                
                # Display recommendations
                st.markdown(f"#### 🌟 Top {len(recommendations)} Similar Restaurants")
                
                for i, (restaurant_name, similarity_score) in enumerate(recommendations):
                    details = engine.get_restaurant_details(restaurant_name)
                    
                    with st.expander(f"#{i+1} - {restaurant_name.title()}", expanded=i < 3):
                        display_restaurant_card(restaurant_name, details, similarity_score)
                        
                        # Additional details in columns
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            if details.get('latitude') and details.get('longitude'):
                                st.write(f"📍 **Coordinates:** {details['latitude']:.4f}, {details['longitude']:.4f}")
                        with detail_col2:
                            if details.get('url'):
                                st.markdown(f"🔗 [Official Website]({details['url']})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        🤖 Powered by Machine Learning • 🍽️ Michelin Restaurant Data • ⭐ TF-IDF Cuisine Analysis
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
