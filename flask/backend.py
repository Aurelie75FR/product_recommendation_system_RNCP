from flask import Flask, jsonify, request
from google.cloud import bigquery
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
client = bigquery.Client()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    # Exemple de requête BigQuery
    query = """
    SELECT asin, AVG(Ratings) AS avg_rating, COUNT(*) AS review_count
    FROM `bot-recommendation.amazonuk_data.combined_data`
    WHERE product_id IN (
        SELECT product_id
        FROM `votre_projet.amazon_data.combined_data`
        WHERE user_id = @user_id
    )
    GROUP BY product_id
    ORDER BY avg_rating DESC
    LIMIT 10
    """
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        recommendations = [dict(row) for row in results]
        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500