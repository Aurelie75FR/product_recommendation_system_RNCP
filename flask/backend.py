#librairies
import os
import logging
import pandas as pd
import numpy as np

from google.cloud import bigquery
from config import Config

# ML model
from utils.recommender import AmazonRecommender



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryClient:
    def __init__(self):
        try:
            # Vérifier si le fichier de credentials existe
            creds_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), Config.GOOGLE_APPLICATION_CREDENTIALS)
            if not os.path.exists(creds_path):
                raise FileNotFoundError(f"Credentials file not found: {creds_path}")
            
            logger.info(f"Using credentials from: {creds_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            
            self.client = bigquery.Client()
            self.project_id = Config.PROJECT_ID
            self.dataset_id = Config.DATASET_ID
            self.recommender = AmazonRecommender()
            self._init_recommender()
            
            logger.info(f"BigQuery client initialized with project: {self.project_id}, dataset: {self.dataset_id}")
            
            # Test de connexion immédiat
            if not self.test_connection():
                raise Exception("Failed to connect to BigQuery")
                
        except Exception as e:
            logger.error(f"Error initializing BigQueryClient: {str(e)}")
            raise

    def test_connection(self):
        """Test simple de la connexion BigQuery"""
        try:
            logger.info("Testing BigQuery connection...")
            query = "SELECT 1 as test"
            result = self.execute_query(query)
            for row in result:
                logger.info(f"Connection test result: {row.test}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


    def execute_query(self, query, params=None):
        """Exécute une requête BigQuery avec des paramètres optionnels"""
        try:
            job_config = bigquery.QueryJobConfig()
            if params:
                job_config.query_parameters = params
            
            logger.info(f"Starting query execution: {query[:100]}...")
            query_job = self.client.query(query, job_config=job_config)
            
            # Augmentation du timeout à 60 secondes
            result = query_job.result(timeout=60)
            logger.info("Query completed successfully")
            return result
            
        except TimeoutError:
            logger.error("Query timed out after 60 seconds")
            raise
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def _init_recommender(self):
        try:
            query = f"""
            SELECT 
                p.asin,  
                p.title,
                p.img_url,
                p.product_url,
                p.price,
                c.category_name as categoryName,
                pf.stars,
                pf.reviews,
                pf.popularity_score,
                pf.value_for_money
            FROM `{self.project_id}.{self.dataset_id}.ProductFeatures` pf
            JOIN `{self.project_id}.{self.dataset_id}.CleanProducts` p ON pf.asin = p.asin
            JOIN `{self.project_id}.{self.dataset_id}.Categories` c ON p.category_id = c.category_id
            WHERE pf.reviews > 100
            """
            result = self.execute_query(query)
            
            # Convertir en DataFrame
            df = pd.DataFrame([dict(row) for row in result])
            print(f"Data columns available: {df.columns.tolist()}")
            
            self.recommender.fit(df)
            logger.info("Recommender model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommender: {str(e)}")
            raise

    def get_popular_products(self, limit=10, category=None, sort_by=None):
        """Récupère les produits les plus populaires avec filtres"""
        try:
            # Définir les préférences de base
            user_prefs = {
                'categories': [category] if category and category != "All categories" 
                            else list(self.recommender.product_data['categoryName'].unique()),
                'min_price': 0,
                'max_price': float('inf'),
                'min_rating': 4.0
            }
            
            # Forcer le mode tri si sort_by est spécifié
            force_sort = sort_by in ["Price: Low to High", "Price: High to Low"]
            
            print(f"Debug - Category: {category}")
            print(f"Debug - Sort by: {sort_by}")
            print(f"Debug - Force sort: {force_sort}")
            
            recommendations = self.recommender.get_personalized_recommendations(
                user_prefs, 
                n=limit,
                sort_by=sort_by,
                force_sort=force_sort  # Nouveau paramètre
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in get_popular_products: {str(e)}")
            raise

    def get_similar_products(self, asin, limit=5):
        """Trouve des produits similaires"""
        recommendations = self.recommender.get_similar_products(asin, n=limit)
        return recommendations .reset_index()

    def get_category_recommendations(self, category_id, limit=5):
        """Récupère les recommandations par catégorie"""
        # D'abord obtenir le nom de la catégorie
        query = f"""
        SELECT category_name 
        FROM `{self.project_id}.{self.dataset_id}.Categories`
        WHERE category_id = @category_id
        """
        params = [bigquery.ScalarQueryParameter("category_id", "STRING", category_id)]
        result = self.execute_query(query, params)
        category_name = next(result).category_name
        
        recommendations = self.recommender.get_category_recommendations(
            category_name, 
            n=limit
        )
        return recommendations.reset_index()