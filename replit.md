# Michelin Restaurant Recommender

## Overview

This is a Streamlit-based web application that provides personalized Michelin restaurant recommendations using machine learning techniques. The system analyzes restaurant data from Michelin Guide and uses content-based filtering with TF-IDF vectorization and cosine similarity to suggest restaurants based on user preferences and restaurant characteristics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Styling**: Custom CSS with futuristic gradient themes and card-based layouts
- **Layout**: Wide layout with collapsible sidebar for optimal user experience
- **Responsive Design**: Card-based restaurant display system with custom styling

### Backend Architecture
- **Core Engine**: Custom `RestaurantRecommendationEngine` class handling all recommendation logic
- **Machine Learning**: Scikit-learn based content filtering using TF-IDF vectorization
- **Data Processing**: Pandas for data manipulation and preprocessing
- **Similarity Matching**: Cosine similarity for restaurant recommendations and fuzzy string matching for search

## Key Components

### 1. Main Application (`app.py`)
- Streamlit web interface with custom CSS styling
- User interaction handling and recommendation display
- Page configuration and layout management

### 2. Recommendation Engine (`utils/recommendation_engine.py`)
- **Data Loading**: Processes three separate CSV files for 1, 2, and 3-star Michelin restaurants
- **Preprocessing**: Data cleaning, missing value handling, and feature engineering
- **ML Pipeline**: TF-IDF vectorization and cosine similarity matrix computation
- **Fuzzy Matching**: Restaurant name search with fuzzy string matching using fuzzywuzzy

### 3. Data Structure
- **Multi-tier System**: Separate datasets for different Michelin star levels (1, 2, 3 stars)
- **Unified Processing**: Combines all star levels into single dataframe with star rating column
- **Feature Engineering**: Creates composite features for similarity computation

## Data Flow

1. **Data Ingestion**: Load three CSV files containing Michelin restaurant data
2. **Data Preprocessing**: Clean data, handle missing values, add star ratings
3. **Feature Engineering**: Create TF-IDF vectors from restaurant characteristics
4. **Similarity Computation**: Build cosine similarity matrix for all restaurants
5. **User Input Processing**: Handle user preferences and search queries
6. **Recommendation Generation**: Apply similarity matching and return ranked results
7. **Result Display**: Present recommendations through Streamlit interface

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms (TF-IDF, cosine similarity)
- **fuzzywuzzy**: Fuzzy string matching for restaurant search

### Data Requirements
- Three CSV files containing Michelin restaurant data:
  - `one-star-michelin-restaurants.csv`
  - `two-stars-michelin-restaurants.csv`
  - `three-stars-michelin-restaurants.csv`

## Deployment Strategy

### Development Environment
- **Platform**: Designed for Replit deployment with Streamlit
- **Structure**: Simple file-based architecture suitable for cloud deployment
- **Dependencies**: All dependencies manageable through pip/requirements.txt

### Production Considerations
- **Scalability**: Content-based filtering scales well with restaurant database size
- **Performance**: Pre-computed similarity matrices for fast recommendation generation
- **Data Updates**: Modular design allows easy addition of new restaurant data

### Key Architectural Decisions

1. **Content-Based Filtering**: Chosen over collaborative filtering due to lack of user interaction data
2. **TF-IDF Vectorization**: Enables semantic similarity matching between restaurants
3. **Streamlit Framework**: Rapid development and deployment with minimal infrastructure requirements
4. **File-Based Data Storage**: Simple CSV storage suitable for static restaurant data
5. **Fuzzy Matching**: Improves user experience by handling typos in restaurant name searches

The system prioritizes simplicity and ease of deployment while maintaining sophisticated recommendation capabilities through proven machine learning techniques.