from google.cloud import bigquery
from config import Config
import logging
import os

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

    def get_product_by_id(self, asin):
        """Récupère un produit par son ASIN"""
        query = f"""
        SELECT 
            p.asin,
            p.title,
            p.img_url,
            p.product_url,
            p.price,
            pf.stars,
            pf.reviews,
            pf.popularity_score
        FROM `{self.project_id}.{self.dataset_id}.ProductFeatures` pf
        JOIN `{self.project_id}.{self.dataset_id}.CleanProducts` p ON pf.asin = p.asin
        WHERE p.asin = @asin
        """
        params = [bigquery.ScalarQueryParameter("asin", "STRING", asin)]
        return self.execute_query(query, params)
    
    def search_products(self, search_term, category_id=None, limit=10):
        """Recherche des produits par terme et catégorie"""
        conditions = ["LOWER(p.title) LIKE CONCAT('%', LOWER(@search_term), '%')"]
        params = [
            bigquery.ScalarQueryParameter("search_term", "STRING", search_term),
            bigquery.ScalarQueryParameter("limit", "INT64", limit)
        ]
        
        if category_id:
            conditions.append("p.category_id = @category_id")
            params.append(bigquery.ScalarQueryParameter("category_id", "STRING", category_id))

        query = f"""
        SELECT 
            p.asin,
            p.title,
            p.img_url,
            p.product_url,
            p.price,
            pf.stars,
            pf.reviews,
            pf.popularity_score,
            pf.value_for_money,
            pf.review_segment,
            p.price_category
        FROM `{self.project_id}.{self.dataset_id}.ProductFeatures` pf
        JOIN `{self.project_id}.{self.dataset_id}.CleanProducts` p 
            ON pf.asin = p.asin
        WHERE {' AND '.join(conditions)}
        ORDER BY pf.popularity_score DESC
        LIMIT @limit
        """
        return self.execute_query(query, params)

    def get_popular_products(self, limit=10):
        """Récupère les produits les plus populaires"""
        query = f"""
        SELECT 
            p.asin,
            p.title,
            p.img_url,
            p.product_url,
            p.price,
            pf.stars,
            pf.reviews,
            pf.popularity_score,
            pf.value_for_money,
            pf.review_segment,
            p.price_category
        FROM `{self.project_id}.{self.dataset_id}.ProductFeatures` pf
        JOIN `{self.project_id}.{self.dataset_id}.CleanProducts` p 
            ON pf.asin = p.asin
        ORDER BY pf.popularity_score DESC
        LIMIT @limit
        """
        params = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        return self.execute_query(query, params)


    def get_popular_products(self, limit=10):
        """Récupère les produits les plus populaires avec diversité maximale"""
        query = f"""
        WITH ProductsWithBrands AS (
            SELECT 
                p.asin,
                p.title,
                p.img_url,
                p.product_url,
                p.price,
                p.category_id,
                c.category_name,
                pf.stars,
                pf.reviews,
                pf.popularity_score,
                pf.value_for_money,
                pf.review_segment,
                p.price_category,
                -- Extrait la marque du titre
                REGEXP_EXTRACT(LOWER(p.title), r'^([a-zA-Z\s]+)') as brand,
                -- Rang dans la catégorie
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_name
                    ORDER BY pf.popularity_score DESC, pf.value_for_money DESC
                ) as category_rank,
                -- Rang par marque
                ROW_NUMBER() OVER (
                    PARTITION BY REGEXP_EXTRACT(LOWER(p.title), r'^([a-zA-Z\s]+)')
                    ORDER BY pf.popularity_score DESC, pf.value_for_money DESC
                ) as brand_rank
            FROM `{self.project_id}.{self.dataset_id}.ProductFeatures` pf
            JOIN `{self.project_id}.{self.dataset_id}.CleanProducts` p ON pf.asin = p.asin
            JOIN `{self.project_id}.{self.dataset_id}.Categories` c ON p.category_id = c.category_id
            WHERE 
                pf.reviews > 1000
                AND pf.stars >= 4.0
                -- Exclure les catégories mal attribuées pour les piles
                AND NOT (
                    LOWER(p.title) LIKE '%batteries%' 
                    AND c.category_name != 'Household Batteries, Chargers & Accessories'
                )
        )
        SELECT 
            asin,
            title,
            img_url,
            product_url,
            price,
            stars,
            reviews,
            popularity_score,
            value_for_money,
            review_segment,
            price_category,
            category_name
        FROM ProductsWithBrands
        WHERE 
            category_rank = 1  -- Meilleur produit de chaque catégorie
            AND brand_rank = 1  -- Meilleur produit de chaque marque
        ORDER BY popularity_score DESC, value_for_money DESC
        LIMIT @limit
        """
        params = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        return self.execute_query(query, params)
    

    def get_category_analytics(self, category_id):
        """Récupère les statistiques d'une catégorie"""
        query = f"""
        SELECT 
            category_id,
            product_count,
            avg_popularity,
            avg_value_for_money,
            avg_stars,
            avg_price,
            avg_reviews,
            bestseller_count
        FROM `{self.project_id}.{self.dataset_id}.ProductAnalytics`
        WHERE category_id = @category_id
        """
        params = [bigquery.ScalarQueryParameter("category_id", "STRING", category_id)]
        return self.execute_query(query, params)