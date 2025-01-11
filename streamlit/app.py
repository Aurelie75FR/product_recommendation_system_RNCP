import streamlit as st
import pandas as pd
import numpy as np
import requests
from api_helper import APIHelper

# Page config
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

# Initialisation des états
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'api' not in st.session_state:
    st.session_state.api = APIHelper(base_url="http://127.0.0.1:5000")

# Test de connexion backend
try:
    response = requests.get("http://127.0.0.1:5000/api/health")
    st.sidebar.success(f"Backend connection OK: {response.status_code}")
except Exception as e:
    st.sidebar.error(f"Backend connection failed: {str(e)}")

# Style CSS global - Déplacé au début et toujours appliqué
st.markdown("""
<style>
    /* Style pour la barre de recherche */
    div[data-testid="stTextInput"] input {
        background-color: #2b2b2b !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Styles des cartes produits */
    .product-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    .product-image {
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .product-image img {
        max-height: 200px;
        width: auto;
        object-fit: contain;
    }

    /* Style des boutons Streamlit */
    button[kind="secondary"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    button[kind="secondary"]:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
    }

    /* Style des liens */
    a {
        color: #4da6ff !important;
        text-decoration: none !important;
    }

    a:hover {
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

def show_product_card(product, idx, is_recommendation=False):
    """Affiche une carte produit"""
    with st.container():
        st.markdown(f"""
            <div class="product-card">
                <div class="product-image">
                    <img src="{product['imgUrl'] if pd.notna(product.get('imgUrl')) else ''}" alt="product image">
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**{product['title'][:100]}...**")
        st.write(f"💰 Price: £{product['price']:.2f}")
        st.write(f"⭐ Rating: {product['stars']:.1f}/5 ({int(product['reviews']):,} reviews)")
        st.write(f"📁 {product['categoryName']}")
        
        col1, col2 = st.columns(2)
        with col1:
            if pd.notna(product.get('productURL')):
                st.markdown(f"[See on Amazon]({product['productURL']})")
        with col2:
            prefix = "rec" if is_recommendation else "prod"
            if st.button("See details", key=f"{prefix}_{idx}"):
                st.session_state.selected_product = product
                if not is_recommendation:
                    st.session_state.current_page = 'detail'
                st.rerun()

def show_product_detail(product):
    """Affiche la page détaillée d'un produit"""
    if st.button("← Back to Result"):
        st.session_state.current_page = 'main'
        # Réinitialiser les filtres de la page détail
        if 'detail_sort_order' in st.session_state:
            del st.session_state.detail_sort_order
        if 'detail_rating_slider' in st.session_state:
            del st.session_state.detail_rating_slider
        if 'detail_min_reviews' in st.session_state:
            del st.session_state.detail_min_reviews
        if 'detail_price_range' in st.session_state:
            del st.session_state.detail_price_range
        st.rerun()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pd.notna(product.get('imgUrl')):
            try:
                st.image(product['imgUrl'], width=300)
            except:
                st.write("🖼️ Image not available")
            
    with col2:
        st.title(product['title'])
        st.write(f"💰 Price: £{product['price']:.2f}")
        st.write(f"⭐ Rating: {product['stars']:.1f}/5 ({int(product['reviews']):,} reviews)")
        st.write(f"📁 Category: {product['categoryName']}")
        
        if pd.notna(product.get('productURL')):
            st.markdown(f"[See on Amazon]({product['productURL']})")

    # Section recommandations
    st.markdown("---")
    st.header("Recommended similar products")
    
    try:
        recommendations = st.session_state.api.get_product_recommendations(product['asin'])
        
        if recommendations:
            # Calcul des prix min et max pour les recommandations
            min_price = round(min(rec['price'] for rec in recommendations), 2)
            max_price = round(max(rec['price'] for rec in recommendations), 2)

            # Filtres dans la sidebar pour les recommandations
            with st.sidebar:
                st.write("Filters")
                
                # Price range slider
                price_range = st.slider(
                    "Price range (£)",
                    min_value=float(min_price),
                    max_value=float(max_price),
                    value=(float(min_price), float(max_price)),
                    key="detail_price_range"
                )
                
                sort_order = st.radio(
                    "Sort by price",
                    ["None", "Price: Low to High", "Price: High to Low"],
                    key="detail_sort_order"
                )
                
                min_rating = st.slider(
                    "Minimum rating",
                    1.0, 5.0, 4.0, 0.5,
                    key="detail_rating_slider"
                )
                
                min_reviews = st.number_input(
                    "Minimum number of reviews",
                    0, 1000000, 100,
                    key="detail_min_reviews"
                )

            # Filtrage avec le nouveau critère de prix
            filtered_recommendations = [
                rec for rec in recommendations
                if (rec['stars'] >= min_rating and 
                    rec['reviews'] >= min_reviews and
                    price_range[0] <= rec['price'] <= price_range[1])
            ]

            if sort_order == "Price: Low to High":
                filtered_recommendations = sorted(filtered_recommendations, 
                                               key=lambda x: x['price'])
            elif sort_order == "Price: High to Low":
                filtered_recommendations = sorted(filtered_recommendations, 
                                               key=lambda x: x['price'], 
                                               reverse=True)

            if filtered_recommendations:
                cols = st.columns(3)
                for idx, rec in enumerate(filtered_recommendations):
                    with cols[idx % 3]:
                        show_product_card(rec, idx, is_recommendation=True)
            else:
                st.warning("No recommendations found matching your criteria.")
        else:
            st.warning("No recommendations found for this product.")
            
    except Exception as e:
        st.error(f"Error loading recommendations: {str(e)}")

# Page principale
if st.session_state.current_page == 'main':
    st.title("🛍️ Recommendation system with Amazon products")
    
    # Sidebar - Filtres
    with st.sidebar:
        st.write("Filters")
        
        categories = ["All categories"] + st.session_state.api.get_categories()
        selected_category = st.selectbox("Select category", categories)
        
        sort_order = st.radio(
            "Sort by price",
            ["None", "Price: Low to High", "Price: High to Low"]
        )
    
    # Barre de recherche
    search_query = st.text_input("🔍 Product search", 
                            value="",
                            placeholder="Search for products...",
                            key="search_input"
    )
    
    try:
        if search_query:
            products = st.session_state.api.search_products(
                query=search_query,
                category=selected_category,
                sort_by=sort_order,
                limit=30
            )
            st.write(f"Results for '{search_query}': {len(products)} products")
        else:
            products = st.session_state.api.get_popular_products(
                limit=30,
                category=selected_category,
                sort_by=sort_order
            )
            st.write("Most popular products:")

        if products:
            min_price = round(min(product['price'] for product in products), 2)
            max_price = round(max(product['price'] for product in products), 2)
        else:
            min_price = 0.0
            max_price = 1000.0

        # Filtres supplémentaires dans la sidebar
        with st.sidebar:
            price_range = st.slider(
                "Price range (£)",
                min_value=float(min_price),
                max_value=float(max_price),
                value=(float(min_price), float(max_price)),
                key="main_price_slider"
            )

            min_rating = st.slider(
                "Minimum rating",
                1.0, 5.0, 4.0, 0.5,
                key="main_rating_slider"
            )
            
            min_reviews = st.number_input(
                "Minimum number of reviews",
                0, 1000000, 100,
                key="main_min_reviews"
            )

        if products:
            filtered_products = [
                product for product in products
                if (price_range[0] <= product['price'] <= price_range[1] and
                    product['stars'] >= min_rating and
                    product['reviews'] >= min_reviews)
            ]
            
            if filtered_products:
                cols = st.columns(3)
                for idx, product in enumerate(filtered_products):
                    with cols[idx % 3]:
                        show_product_card(product, idx)
            else:
                st.warning("No products found matching your criteria.")
        else:
            st.warning("No products found.")
            
    except Exception as e:
        st.error(f"Error loading products: {str(e)}")

# Page détail produit
elif st.session_state.current_page == 'detail' and st.session_state.selected_product is not None:
    show_product_detail(st.session_state.selected_product)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ By Aurélie")