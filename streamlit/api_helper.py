import requests
import pandas as pd

class APIHelper:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        """Initialize API helper with base URL"""
        self._base_url = base_url.rstrip('/')  # Utilisez un underscore pour indiquer que c'est une variable d'instance
        # Test de connexion
        try:
            health_check = requests.get(f"{self._base_url}/api/health", timeout=5)
            if health_check.status_code != 200:
                print(f"Warning: API health check failed with status {health_check.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to API at {self._base_url}")
        except Exception as e:
            print(f"Warning: API health check failed: {str(e)}")

    # def __normalize_product(self, product):
    #     """Normalise les champs du produit pour correspondre à l'interface Streamlit"""
    #     try:
    #         # Vérification de la présence des champs requis
    #         required_fields = ['title', 'img_url', 'product_url', 'price', 'categoryName', 'stars', 'reviews', 'asin']
    #         for field in required_fields:
    #             if field not in product:
    #                 print(f"Warning: Missing field '{field}' in product data")
    #                 # Définir des valeurs par défaut selon le champ manquant
    #                 if field == 'title':
    #                     product[field] = "Unknown Product"
    #                 elif field in ['img_url', 'product_url']:
    #                     product[field] = ""
    #                 elif field in ['price', 'stars']:
    #                     product[field] = 0.0
    #                 elif field == 'reviews':
    #                     product[field] = 0
    #                 elif field == 'categoryName':
    #                     product[field] = "Uncategorized"
    #                 elif field == 'asin':
    #                     product[field] = "unknown"

    #         return {
    #             'title': str(product['title']),
    #             'imgUrl': str(product['img_url']),
    #             'productURL': str(product['product_url']),
    #             'price': float(product['price'] or 0.0),
    #             'categoryName': str(product['categoryName']),
    #             'stars': float(product['stars'] or 0.0),
    #             'reviews': int(product['reviews'] or 0),
    #             'asin': str(product['asin'])
    #         }
    #     except Exception as e:
    #         print(f"Error normalizing product: {str(e)}")
    #         print(f"Product data: {product}")  # Pour le debug
    #         raise Exception(f"Failed to normalize product: {str(e)}")

    def __normalize_product(self, product):
        """Normalise les champs du produit pour correspondre à l'interface Streamlit"""
        try:
            # Debug: afficher toutes les clés disponibles
            print(f"Available keys in product: {list(product.keys())}")
            
            # Vérification et récupération de l'ASIN
            asin = product.get('asin')
            if not asin:
                raise ValueError(f"Missing required field 'asin' in product data. Available fields: {list(product.keys())}")
                
            return {
                'title': str(product.get('title', 'Unknown Title')),
                'imgUrl': str(product.get('img_url', '')),
                'productURL': str(product.get('product_url', '')),
                'price': float(product.get('price', 0.0)),
                'categoryName': str(product.get('categoryName', 'Uncategorized')),
                'stars': float(product.get('stars', 0.0)),
                'reviews': int(product.get('reviews', 0)),
                'asin': str(asin)
            }
        except Exception as e:
            print(f"Error normalizing product: {str(e)}")
            print(f"Product data: {product}")
            raise ValueError(f"Failed to normalize product: {str(e)}")

    def get_popular_products(self, limit=30):
        try:
            url = f"{self._base_url}/api/products/popular"
            print(f"Requesting: {url}")
            
            response = requests.get(url, params={"limit": limit}, timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response data: {data}")  # Debug
                
                if "products" not in data:
                    print("Warning: 'products' key not found in response")
                    return []
                    
                normalized_products = []
                for product in data["products"]:
                    try:
                        normalized = self.__normalize_product(product)
                        normalized_products.append(normalized)
                    except Exception as e:
                        print(f"Failed to normalize product: {str(e)}")
                        continue
                        
                return normalized_products
                
            print(f"Error: API returned status code {response.status_code}")
            return []
        except Exception as e:
            print(f"Error fetching popular products: {str(e)}")
            return []

    def get_product_recommendations(self, asin, limit=5):
        """Get recommendations for a specific product"""
        try:
            url = f"{self._base_url}/api/products/{asin}/recommendations"
            print(f"Requesting recommendations: {url}")
            
            response = requests.get(url, params={"limit": limit}, timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "recommendations" in data:
                    return [self.__normalize_product(p) for p in data["recommendations"]]
                print("Warning: 'recommendations' key not found in response")
                return []
            print(f"Error: API returned status code {response.status_code}")
            return []
        except Exception as e:
            print(f"Error fetching recommendations: {str(e)}")
            return []