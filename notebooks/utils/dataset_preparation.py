import pandas as pd
import numpy as np
from collections import Counter

class DataPreparation:
    """
    Class to prepare data for Flourish and generate a dataset for wordcloud
    """
    def __init__(self, df):
        self.df = df
    
    def generate_category_stats(self, output_dir='../data/dashboard_data/'):
        """
        Generates CSV files for different Flourish visualizations
        """
        # Create the directory if necessary
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribution of products by category
        category_counts = self.df['categoryName'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Number_of_Products']
        category_counts.to_csv(f'{output_dir}category_distribution.csv', index=False)
        
        # 2. Average price by category
        price_by_category = self.df.groupby('categoryName')['price'].agg(['mean', 'min', 'max']).reset_index()
        price_by_category.columns = ['Category', 'Average_Price', 'Min_Price', 'Max_Price']
        price_by_category = price_by_category.round(2)
        price_by_category.to_csv(f'{output_dir}price_analysis.csv', index=False)
        
        # 3. Ratings analysis
        ratings_by_category = self.df.groupby('categoryName').agg({
            'stars': ['mean', 'count'],
            'isBestSeller': 'sum'
        }).reset_index()
        ratings_by_category.columns = ['Category', 'Average_Rating', 'Number_of_Reviews', 'Number_of_Bestsellers']
        ratings_by_category = ratings_by_category.round(2)
        ratings_by_category.to_csv(f'{output_dir}ratings_analysis.csv', index=False)
        
        # 4. Temporal analysis if there is a date column
        if 'date' in self.df.columns:
            time_series = self.df.groupby(['date', 'categoryName']).size().reset_index()
            time_series.columns = ['Date', 'Category', 'Number_of_Products']
            time_series.to_csv(f'{output_dir}time_series.csv', index=False)
    
    def generate_wordcloud_data(self, output_file='../data/dashboard_data/wordcloud_data.csv'):
        """
        Generates a dataset for the wordcloud based on product titles
        """
        # Combine all titles
        all_words = ' '.join(self.df['title'].astype(str)).lower()
        
        # List of words to exclude (stopwords)
        stopwords = set(['the', 'a', 'an', 'and', 'of', 'for', 'to', 'in', 'on', 'at', 'by'])
        
        # Clean and count the words
        words = [word for word in all_words.split() if len(word) > 2 and word not in stopwords]
        word_freq = Counter(words)
        
        # Create a DataFrame for the wordcloud
        wordcloud_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['frequency'])
        wordcloud_df.index.name = 'word'
        wordcloud_df = wordcloud_df.reset_index()
        wordcloud_df = wordcloud_df.sort_values('frequency', ascending=False)
        
        # Save the file
        wordcloud_df.to_csv(output_file, index=False)
