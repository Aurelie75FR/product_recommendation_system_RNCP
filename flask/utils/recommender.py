import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning)

class AmazonRecommender:
    """
    Amazon product recommendation system
    """
    def __init__(self):
        self.knn_model = None
        self.scaler = MinMaxScaler()
        self.product_features = None
        self.product_data = None
    
    def _calculate_category_similarity(self, cat1, cat2):
        """
        Calculates the similarity between two categories based on business rules
        """
        electronics_categories = {
            'Hi-Fi Speakers', 'PC & Video Games', 'PC Gaming Accessories',
            'Headphones', 'Home Audio & Theater', 'TV & Home Cinema',
            'Home Entertainment', 'Electrical', 'Consumer Electronics'
        }
        
        home_categories = {
            'Home & Kitchen', 'Furniture', 'Home Entertainment Furniture',
            'Home Improvement', 'Home Accessories', 'Home Storage'
        }
        
        if cat1 == cat2:
            return 1.0
        elif cat1 in electronics_categories and cat2 in electronics_categories:
            if 'Speakers' in cat1 and 'Speakers' in cat2:
                return 0.8
            return 0.6
        elif cat1 in home_categories and cat2 in home_categories:
            return 0.6
        else:
            return 0.3
    
    def create_product_features(self, df):
        """
        Creates features with adaptive weightings
        """
        feature_df = pd.DataFrame()
        
        # Standardised numerical features
        feature_df['price_norm'] = np.log1p(df['price']) / np.log1p(df['price'].max())
        feature_df['stars_norm'] = df['stars'] / 5
        feature_df['reviews_norm'] = np.log1p(df['reviews']) / np.log1p(df['reviews'].max())
        
        # Price segmentation
        price_quantiles = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        price_dummies = pd.get_dummies(price_quantiles, prefix='price')
        
        # Segmentation of ratings
        rating_cats = pd.cut(df['stars'], 
                            bins=[0, 3.5, 4.0, 4.5, 5.0], 
                            labels=['low', 'medium', 'high', 'very_high'])
        rating_dummies = pd.get_dummies(rating_cats, prefix='rating')
        
        # Main categories
        top_categories = df['categoryName'].value_counts().nlargest(50).index
        df_filtered = df.copy()
        df_filtered.loc[~df_filtered['categoryName'].isin(top_categories), 'categoryName'] = 'Other'
        category_dummies = pd.get_dummies(df_filtered['categoryName'], prefix='category')
        
        self.product_features = pd.concat([
            feature_df,
            price_dummies,
            rating_dummies,
            category_dummies
        ], axis=1)
        
        self.product_features = pd.DataFrame(
            self.scaler.fit_transform(self.product_features),
            columns=self.product_features.columns
        )
        
        return self.product_features


    def get_similar_products(self, product_id, n=5):
        try:
            # Vérifier si le product_id existe dans les données
            if product_id not in self.product_data['asin'].values:
                print(f"Product ID {product_id} not found in data")
                return pd.DataFrame()
                
            # Obtenir le produit original
            original = self.product_data[self.product_data['asin'] == product_id].iloc[0]
            
            # Filtrage initial
            similar_products = self.product_data[
                (self.product_data['asin'] != product_id) &
                (self.product_data['categoryName'] == original['categoryName']) &
                (self.product_data['reviews'] > 1000) &
                (self.product_data['stars'] >= 4.0) &
                (self.product_data['price'] >= original['price'] * 0.5) &
                (self.product_data['price'] <= original['price'] * 2.0)
            ].copy()

            if len(similar_products) == 0:
                print(f"No similar products found for {product_id}")
                return pd.DataFrame()

            # Extraction de la marque
            similar_products['brand'] = similar_products['title'].str.extract(r'^([A-Za-z]+)')
            
            # Score de similarité de prix
            similar_products['price_ratio'] = similar_products['price'] / original['price']
            similar_products['price_score'] = 1 - np.abs(np.log(similar_products['price_ratio']))

            # Calcul des scores pour le classement
            similar_products['final_score'] = (
                similar_products['price_score'].clip(0, 1) * 0.3 +
                (similar_products['stars'] / 5) * 0.4 +
                (np.log1p(similar_products['reviews']) / 
                np.log1p(similar_products['reviews'].max())) * 0.3
            ).clip(0, 1)
            
            # Sélection des meilleurs produits
            result = similar_products.nlargest(n, 'final_score')
            
            # Assurez-vous que toutes les colonnes requises sont présentes
            required_columns = [
                'asin', 'title', 'categoryName', 'price', 'stars', 
                'reviews', 'img_url', 'product_url', 'final_score'
            ]
            
            missing_columns = [col for col in required_columns if col not in result.columns]
            if missing_columns:
                print(f"Missing columns in result: {missing_columns}")
                
            return result[required_columns]
                
        except Exception as e:
            print(f"Erreur dans get_similar_products: {str(e)}")
            print(f"Product ID: {product_id}")
            print(f"Available columns: {self.product_data.columns.tolist()}")
            return pd.DataFrame()
    
    def get_category_recommendations(self, category, n=5):
        """
        Recommends the best products in a category
        """
        category_products = self.product_data[
            self.product_data['categoryName'] == category
        ].copy()
        
        if len(category_products) == 0:
            return pd.DataFrame()
        
        category_products['value_score'] = (
            category_products['stars'] / 5 * 0.4 +
            (1 - category_products['price'] / category_products['price'].max()) * 0.3 +
            (np.log1p(category_products['reviews']) / 
             np.log1p(category_products['reviews'].max())) * 0.3
        )
        
        return category_products.nlargest(n, 'value_score')[[
            'title', 'categoryName', 'price', 'stars', 'value_score'
        ]]
    
    
    def get_personalized_recommendations(self, user_prefs, n=5, sort_by=None, force_sort=False):
        """Recommandations personnalisées avec diversité et tri par prix"""
        try:
            # Filtrage initial
            mask = (
                self.product_data['categoryName'].isin(user_prefs['categories']) &
                (self.product_data['price'] >= user_prefs['min_price']) &
                (self.product_data['price'] <= user_prefs['max_price']) &
                (self.product_data['stars'] >= user_prefs['min_rating']) &
                (self.product_data['reviews'] > 1000)
            )
            
            candidates = self.product_data[mask].copy()
            
            if len(candidates) == 0:
                print("No products found matching the criteria")
                return pd.DataFrame()

            # Si force_sort est True ou si on a une catégorie spécifique, appliquer le tri directement
            if force_sort:
                if sort_by == "Price: Low to High":
                    print("Sorting by price: low to high")
                    return candidates.sort_values('price', ascending=True).head(n)
                elif sort_by == "Price: High to Low":
                    print("Sorting by price: high to low")
                    return candidates.sort_values('price', ascending=False).head(n)

            # Pour All categories sans tri forcé, assurer la diversité
            print("Applying diversity logic")
            
            # Calculer le score pour tous les produits
            candidates['score'] = (
                candidates['stars'] / 5 * 0.4 +
                (np.log1p(candidates['reviews']) / 
                np.log1p(candidates['reviews'].max())) * 0.6
            )
            
            # Sélectionner les meilleures catégories
            top_categories = candidates['categoryName'].value_counts().head(6).index
            
            # Sélectionner les meilleurs produits de chaque catégorie
            diverse_products = pd.DataFrame()
            products_per_category = max(1, n // len(top_categories))
            
            for category in top_categories:
                cat_products = candidates[candidates['categoryName'] == category]
                if not cat_products.empty:
                    cat_selection = cat_products.nlargest(products_per_category, 'score')
                    diverse_products = pd.concat([diverse_products, cat_selection])
            
            # Si on a un tri spécifié, l'appliquer sur les produits diversifiés
            if sort_by:
                if sort_by == "Price: Low to High":
                    diverse_products = diverse_products.sort_values('price', ascending=True)
                elif sort_by == "Price: High to Low":
                    diverse_products = diverse_products.sort_values('price', ascending=False)
            
            return diverse_products.head(n)
                
        except Exception as e:
            print(f"Error in get_personalized_recommendations: {str(e)}")
            return pd.DataFrame()

    def fit(self, df, verbose=True):
        """
        Drives the recommendation system
        """
        if verbose:
            print("Creating features...")
        
        # Garder une copie des données originales SANS définir d'index
        self.product_data = df.copy()
        
        # Créer les features pour le KNN
        feature_df = self.create_product_features(df)
        
        if verbose:
            print("Training the KNN model...")
            print(f"Available columns: {self.product_data.columns.tolist()}")
        
        self.knn_model = NearestNeighbors(
            n_neighbors=50,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        self.knn_model.fit(feature_df)
        
        if verbose:
            print("Training completed!")
            
        return self