# flask/app.py
from flask import Flask, jsonify, request
from config import Config
from backend import BigQueryClient
import logging
import pandas as pd
import numpy as np



# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config.from_object(Config)
db = BigQueryClient()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier la santé de l'API"""
    try:
        # Vérifie à la fois BigQuery et le modèle de recommandation
        db.execute_query("SELECT 1")
        model_status = hasattr(db.recommender, 'product_data') and db.recommender.product_data is not None
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "model": "loaded" if model_status else "not loaded"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/products/popular', methods=['GET'])
def get_popular_products():
    try:
        limit = request.args.get('limit', default=10, type=int)
        category = request.args.get('category', default=None, type=str)
        sort_by = request.args.get('sort_by', default=None, type=str)
        
        print(f"Received request - Category: {category}, Sort by: {sort_by}")
        
        user_prefs = {
            'categories': [category] if category and category != "All categories" 
                         else db.recommender.product_data['categoryName'].unique(),
            'min_price': 0.5,
            'max_price': float('inf'),
            'min_rating': 4.0
        }
        
        result = db.recommender.get_personalized_recommendations(
            user_prefs,
            n=limit,
            sort_by=sort_by
        )
        
        if result.empty:
            return jsonify({"error": "No products found"}), 404
            
        products = result.to_dict('records')
        return jsonify({
            "count": len(products),
            "products": products
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching popular products: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/products/<asin>/recommendations', methods=['GET'])
def get_product_recommendations(asin):
    """Recommandations de produits similaires"""
    try:
        limit = request.args.get('limit', default=5, type=int)
        
        recommendations = db.recommender.get_similar_products(asin, n=limit)
        
        if recommendations.empty:
            return jsonify({"error": "No recommendations found"}), 404
            
        # S'assurer que l'index (ASIN) est inclus dans les données
        recommendations = recommendations.reset_index()  # Ceci va inclure l'index (ASIN) comme colonne
        products = recommendations.to_dict('records')
        
        return jsonify({
            "asin": asin,
            "recommendations": products,
            "count": len(products)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_product_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories/<category_id>/recommendations', methods=['GET'])
def get_category_recommendations(category_id):
    """Recommandations par catégorie"""
    try:
        limit = request.args.get('limit', default=5, type=int)
        recommendations = db.get_category_recommendations(category_id, limit)
        
        if recommendations.empty:
            return jsonify({"error": "No recommendations found"}), 404
            
        products = recommendations.to_dict('records')
        return jsonify({
            "category_id": category_id,
            "recommendations": products,
            "count": len(products)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_category_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/products/search', methods=['GET'])
def search_products():
    """Recherche de produits avec recommandations personnalisées"""
    try:
        query = request.args.get('q', '')
        category = request.args.get('category', '')
        min_price = request.args.get('min_price', default=0, type=float)
        max_price = request.args.get('max_price', default=float('inf'), type=float)
        min_rating = request.args.get('min_rating', default=4.0, type=float)
        limit = request.args.get('limit', default=10, type=int)
        
        if not query and not category:
            return jsonify({"error": "Search query or category required"}), 400
            
        # Filtrer les produits selon les critères
        mask = pd.Series(True, index=db.recommender.product_data.index)
        
        if query:
            mask &= db.recommender.product_data['title'].str.contains(query, case=False)
        if category:
            mask &= db.recommender.product_data['categoryName'] == category
            
        user_prefs = {
            'categories': [category] if category else db.recommender.product_data['categoryName'].unique(),
            'min_price': min_price,
            'max_price': max_price,
            'min_rating': min_rating
        }
        
        filtered_data = db.recommender.product_data[mask].copy()
        
        if filtered_data.empty:
            return jsonify({"error": "No products found"}), 404
            
        # Calculer les scores pour le tri
        filtered_data['search_score'] = (
            filtered_data['stars'] / 5 * 0.4 +
            (np.log1p(filtered_data['reviews']) / 
             np.log1p(filtered_data['reviews'].max())) * 0.6
        )
        
        results = filtered_data.nlargest(limit, 'search_score')
        products = results.to_dict('records')
        
        return jsonify({
            "query": query,
            "category": category,
            "products": products,
            "count": len(products)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in search_products: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Récupère toutes les catégories disponibles"""
    try:
        categories = sorted(db.recommender.product_data['categoryName'].unique().tolist())
        return jsonify({
            "categories": categories,
            "count": len(categories)
        }), 200
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG)