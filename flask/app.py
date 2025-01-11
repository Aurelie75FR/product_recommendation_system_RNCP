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
    """Recherche de produits par titre avec gestion améliorée des correspondances"""
    try:
        query = request.args.get('q', '').strip()
        category = request.args.get('category', default=None, type=str)
        sort_by = request.args.get('sort_by', default=None, type=str)
        min_rating = request.args.get('min_rating', default=4.0, type=float)
        min_price = request.args.get('min_price', default=0.0, type=float)
        max_price = request.args.get('max_price', default=float('inf'), type=float)
        min_reviews = request.args.get('min_reviews', default=100, type=int)
        limit = request.args.get('limit', default=30, type=int)
        
        print(f"Search params: query='{query}', category={category}, sort_by={sort_by}")

        # Préparation des données
        data = db.recommender.product_data.copy()
        
        # Normalisation de la requête et création de variations
        query_variations = set()
        query_lower = query.lower()
        query_variations.add(query_lower)
        
        # Ajout des variations communes
        replacements = {
            'playstation': 'ps',
            'ps5': 'playstation 5',
            'ps4': 'playstation 4',
            'ps3': 'playstation 3'
        }
        
        for old, new in replacements.items():
            if old in query_lower:
                query_variations.add(query_lower.replace(old, new))
            if new in query_lower:
                query_variations.add(query_lower.replace(new, old))
        
        # Calcul des scores
        data['search_score'] = 0.0
        
        # 1. Correspondance exacte du titre complet
        exact_matches = data['title'].str.lower().isin([q for q in query_variations])
        data.loc[exact_matches, 'search_score'] += 100.0
        
        # 2. Correspondance du début du titre
        for q in query_variations:
            starts_with = data['title'].str.lower().str.startswith(q)
            data.loc[starts_with, 'search_score'] += 50.0
        
        # 3. Correspondances de mots exacts
        for q in query_variations:
            words = q.split()
            for word in words:
                if len(word) >= 2:  # Ignorer les mots trop courts
                    word_match = data['title'].str.lower().str.contains(
                        fr'\b{word}\b',
                        regex=True,
                        na=False
                    )
                    data.loc[word_match, 'search_score'] += 10.0
        
        # Filtrer les résultats avec un score positif
        results = data[data['search_score'] > 0].copy()
        
        # Appliquer les filtres
        mask = (
            (results['price'] >= min_price) &
            (results['price'] <= max_price) &
            (results['stars'] >= min_rating) &
            (results['reviews'] >= min_reviews)
        )
        
        if category and category != "All categories":
            mask &= results['categoryName'] == category
            
        results = results[mask]
        
        if len(results) == 0:
            return jsonify({
                "query": query,
                "count": 0,
                "products": []
            }), 200
        
        # Score final combinant pertinence et popularité
        results['final_score'] = (
            results['search_score'] * 0.7 +
            (results['stars'] / 5) * 15 +
            (np.log1p(results['reviews']) / np.log1p(results['reviews'].max())) * 15
        )
        
        # Tri des résultats
        if sort_by == "Price: Low to High":
            sorted_results = results.sort_values(['price', 'final_score'], 
                                              ascending=[True, False])
        elif sort_by == "Price: High to Low":
            sorted_results = results.sort_values(['price', 'final_score'], 
                                              ascending=[False, False])
        else:
            sorted_results = results.sort_values('final_score', ascending=False)
        
        final_results = sorted_results.head(limit)
        
        print(f"Found {len(final_results)} results after filtering and sorting")
        
        products = final_results.to_dict('records')
        
        return jsonify({
            "query": query,
            "count": len(products),
            "products": products
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