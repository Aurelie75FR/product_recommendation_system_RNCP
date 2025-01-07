# flask/app.py
from flask import Flask, jsonify, request
from config import Config
from backend import BigQueryClient
import os
import logging

app = Flask(__name__)
app.config.from_object(Config)
db = BigQueryClient()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier la santé de l'API"""
    try:
        logger.info("Performing health check...")
        
        # Test de connexion à BigQuery
        db.test_connection()
        
        response = {
            "status": "healthy",
            "database": "connected",
            "config": {
                "project_id": Config.PROJECT_ID,
                "dataset_id": Config.DATASET_ID,
                "credentials_file": os.path.basename(Config.GOOGLE_APPLICATION_CREDENTIALS)
            }
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "config": {
                "project_id": Config.PROJECT_ID,
                "dataset_id": Config.DATASET_ID,
                "credentials_file": os.path.basename(Config.GOOGLE_APPLICATION_CREDENTIALS)
            }
        }), 500

@app.route('/api/products/popular', methods=['GET'])
def get_popular_products():
    """Récupérer les produits les plus populaires"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        logger.info(f"Fetching {limit} popular products...")
        
        results = db.get_popular_products(limit)
        products = [dict(row) for row in results]
        
        logger.info(f"Successfully fetched {len(products)} products")
        return jsonify({"products": products}), 200
        
    except TimeoutError:
        error_msg = "Request timed out"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 504
    except Exception as e:
        error_msg = f"Error fetching popular products: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/api/products/search', methods=['GET'])
def search_products():
    """Rechercher des produits"""
    try:
        search_term = request.args.get('q', '')
        category_id = request.args.get('category_id')
        limit = request.args.get('limit', default=10, type=int)
        
        if not search_term:
            return jsonify({"error": "Search term is required"}), 400
            
        results = db.search_products(search_term, category_id, limit)
        products = [dict(row) for row in results]
        return jsonify({
            "query": search_term,
            "category_id": category_id,
            "count": len(products),
            "products": products
        }), 200
    except Exception as e:
        logger.error(f"Error searching products: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/products/<asin>', methods=['GET'])
def get_product(asin):
    """Récupérer les détails d'un produit"""
    try:
        results = db.get_product_by_id(asin)
        products = [dict(row) for row in results]
        if not products:
            return jsonify({"error": "Product not found"}), 404
        return jsonify({"product": products[0]}), 200
    except Exception as e:
        logger.error(f"Error fetching product {asin}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/products/<asin>/recommendations', methods=['GET'])
def get_product_recommendations(asin):
    """Récupérer les recommandations pour un produit"""
    try:
        limit = request.args.get('limit', default=5, type=int)
        results = db.get_similar_products(asin, limit)
        recommendations = [dict(row) for row in results]
        return jsonify({
            "asin": asin,
            "count": len(recommendations),
            "recommendations": recommendations
        }), 200
    except Exception as e:
        logger.error(f"Error fetching recommendations for {asin}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories/<category_id>', methods=['GET'])
def get_category_analytics(category_id):
    """Récupérer les statistiques d'une catégorie"""
    try:
        results = db.get_category_stats(category_id)
        stats = [dict(row) for row in results]
        if not stats:
            return jsonify({"error": "Category not found"}), 404
        return jsonify({"category_stats": stats[0]}), 200
    except Exception as e:
        logger.error(f"Error fetching category stats for {category_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG)