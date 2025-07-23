import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
import os

class RestaurantRecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine"""
        self.res_df = None
        self.similarity_df = None
        self.vectorizer = None
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and process the restaurant data"""
        try:
            # Load the three CSV files
            data_dir = "data"
            star1 = pd.read_csv(os.path.join(data_dir, "one-star-michelin-restaurants.csv"))
            star2 = pd.read_csv(os.path.join(data_dir, "two-stars-michelin-restaurants.csv"))
            star3 = pd.read_csv(os.path.join(data_dir, "three-stars-michelin-restaurants.csv"))
            
            # Add star rating column
            star1['stars'] = 1
            star2['stars'] = 2
            star3['stars'] = 3
            
            # Concatenate all data
            self.res_df = pd.concat([star1, star2, star3], ignore_index=True)
            
            # Data cleaning and preprocessing
            self.preprocess_data()
            
            # Build similarity matrix
            self.build_similarity_matrix()
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self):
        """Clean and preprocess the restaurant data"""
        # Drop zipCode if it exists
        if 'zipCode' in self.res_df.columns:
            self.res_df.drop(columns=['zipCode'], inplace=True)
        
        # Fill missing values
        self.res_df['city'] = self.res_df['city'].fillna("Unknown")
        self.res_df['price'] = self.res_df['price'].fillna("Unknown")
        self.res_df['region'] = self.res_df['region'].fillna("Unknown")
        self.res_df['cuisine'] = self.res_df['cuisine'].fillna("Unknown")
        
        # Normalize text fields
        self.res_df['name'] = self.res_df['name'].str.lower().str.strip()
        self.res_df['cuisine'] = self.res_df['cuisine'].str.lower().str.strip()
        
        # Remove duplicates based on name
        self.res_df = self.res_df.drop_duplicates(subset=['name'], keep='first')
        self.res_df = self.res_df.reset_index(drop=True)
    
    def build_similarity_matrix(self):
        """Build the TF-IDF similarity matrix"""
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        # Fit and transform cuisine data
        cuisine_matrix = self.vectorizer.fit_transform(self.res_df['cuisine'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(cuisine_matrix)
        
        # Create similarity DataFrame
        self.similarity_df = pd.DataFrame(
            cosine_sim, 
            index=self.res_df['name'], 
            columns=self.res_df['name']
        )
    
    def find_closest_match(self, restaurant_name):
        """Find the closest matching restaurant name using fuzzy matching"""
        restaurant_name = restaurant_name.lower().strip()
        
        # Try exact match first
        if restaurant_name in self.similarity_df.index:
            return restaurant_name
        
        # Use fuzzy matching
        closest_match = process.extractOne(
            restaurant_name, 
            self.similarity_df.index,
            scorer=fuzz.ratio
        )
        
        if closest_match and closest_match[1] >= 60:  # 60% similarity threshold
            return closest_match[0]
        
        return None
    
    def recommend_restaurant(self, restaurant_name, top_n=5):
        """Get restaurant recommendations based on cuisine similarity"""
        matched_name = self.find_closest_match(restaurant_name)
        
        if not matched_name:
            return f"Restaurant '{restaurant_name}' not found! Please try a different name or check the spelling."
        
        if matched_name not in self.similarity_df.index:
            return f"Restaurant '{matched_name}' not found in similarity matrix!"
        
        # Get similarity scores and sort
        similarities = self.similarity_df[matched_name].sort_values(ascending=False)
        
        # Exclude the restaurant itself and get top N
        recommendations = similarities.iloc[1:top_n+1]
        
        # Return list of (restaurant_name, similarity_score) tuples
        return [(name, score) for name, score in recommendations.items()]
    
    def get_restaurant_details(self, restaurant_name):
        """Get detailed information about a restaurant"""
        restaurant_data = self.res_df[self.res_df['name'] == restaurant_name]
        
        if restaurant_data.empty:
            return {}
        
        details = restaurant_data.iloc[0].to_dict()
        return details
    
    def get_all_restaurant_names(self):
        """Get all restaurant names for autocomplete"""
        return self.res_df['name'].tolist()
    
    def get_restaurant_suggestions(self, partial_name, limit=10):
        """Get restaurant name suggestions for autocomplete"""
        partial_name = partial_name.lower().strip()
        
        # Find restaurants that contain the partial name
        suggestions = self.res_df[
            self.res_df['name'].str.contains(partial_name, case=False, na=False)
        ]['name'].tolist()
        
        if not suggestions:
            # Use fuzzy matching for suggestions
            fuzzy_matches = process.extract(
                partial_name, 
                self.res_df['name'].tolist(), 
                limit=limit,
                scorer=fuzz.partial_ratio
            )
            suggestions = [match[0] for match in fuzzy_matches if match[1] >= 50]
        
        return suggestions[:limit]
    
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            'total_restaurants': len(self.res_df),
            'unique_cuisines': self.res_df['cuisine'].nunique(),
            'unique_cities': self.res_df['city'].nunique(),
            'unique_countries': self.res_df['region'].nunique(),
            'one_star': len(self.res_df[self.res_df['stars'] == 1]),
            'two_star': len(self.res_df[self.res_df['stars'] == 2]),
            'three_star': len(self.res_df[self.res_df['stars'] == 3])
        }
    
    def search_by_cuisine(self, cuisine_type, limit=10):
        """Search restaurants by cuisine type"""
        cuisine_type = cuisine_type.lower().strip()
        
        matching_restaurants = self.res_df[
            self.res_df['cuisine'].str.contains(cuisine_type, case=False, na=False)
        ]
        
        return matching_restaurants.head(limit)
    
    def search_by_location(self, location, limit=10):
        """Search restaurants by city or region"""
        location = location.lower().strip()
        
        matching_restaurants = self.res_df[
            (self.res_df['city'].str.contains(location, case=False, na=False)) |
            (self.res_df['region'].str.contains(location, case=False, na=False))
        ]
        
        return matching_restaurants.head(limit)
