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

try:
    response = requests.get("http://127.0.0.1:5000/api/health")
    st.sidebar.success(f"Backend connection OK: {response.status_code}")
except Exception as e:
    st.sidebar.error(f"Backend connection failed: {str(e)}")

# Init
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'api' not in st.session_state:
    st.session_state.api = APIHelper(base_url="http://127.0.0.1:5000")

# Style CSS global
product_container_style = """
    <style>
        .product-card {
            height: 300px;
            padding: 1rem;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.05);
            margin-bottom: 1rem;
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
    </style>
"""

def show_product_detail(product):
    st.write("Debug - Product data:", product)  # Pour voir les données du produit
    print("Debug - ASIN:", product.get('asin'))  # Pour voir l'ASIN dans les logs
    """render product page details"""
    if st.button("← Back to Result"):
        st.session_state.current_page = 'main'
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

    # Recommendations section
    st.markdown("---")
    st.header("Recommended similar products")
    st.markdown(product_container_style, unsafe_allow_html=True)
    
    try:
        recommendations = st.session_state.api.get_product_recommendations(product['asin'])
        
        if recommendations:
            cols = st.columns(3)
            for idx, rec in enumerate(recommendations):
                with cols[idx % 3]:
                    st.markdown("""
                        <div class="product-card">
                            <div class="product-image">
                                <img src="{}" alt="product image">
                            </div>
                        </div>
                    """.format(rec['imgUrl'] if pd.notna(rec.get('imgUrl')) else ""), 
                    unsafe_allow_html=True)
                    
                    st.markdown(f"**{rec['title'][:100]}...**")
                    st.write(f"💰 Price: £{rec['price']:.2f}")
                    st.write(f"⭐ Rating: {rec['stars']:.1f}/5 ({int(rec['reviews']):,} reviews)")
                    st.write(f"📁 {rec['categoryName']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if pd.notna(rec.get('productURL')):
                            st.markdown(f"[See on Amazon]({rec['productURL']})")
                    with col2:
                        if st.button("See details", key=f"rec_{idx}"):
                            st.session_state.selected_product = rec
                            st.rerun()
        else:
            st.warning("No recommendations found for this product.")
            
    except Exception as e:
        st.error(f"Error loading recommendations: {str(e)}")

# Main page
if st.session_state.current_page == 'main':
    st.title("🛍️ Recommendation system with Amazon products")
    
    # Search bar
    search_query = st.text_input("🔍 Product search", "")
    
    # Sidebar filters
    st.sidebar.write("Filters")
    price_range = st.sidebar.slider(
        "Price range (£)",
        min_value=0.0,
        max_value=1000.0,
        value=(0.0, 300.0)
    )
    
    min_rating = st.sidebar.slider(
        "Minimum rating",
        1.0, 5.0, 4.0, 0.5
    )
    
    min_reviews = st.sidebar.number_input(
        "Minimum number of reviews",
        0, 1000000, 100
    )
    
    # Get and display products
    try:
        if search_query:
            products = st.session_state.api.search_products(
                query=search_query,
                limit=30
            )
            st.write(f"Results for '{search_query}': {len(products)} products")
        else:
            products = st.session_state.api.get_popular_products(limit=30)
            st.write("Most popular products:")

        if products:
            st.markdown(product_container_style, unsafe_allow_html=True)
            cols = st.columns(3)
            
            for idx, product in enumerate(products):
                if (price_range[0] <= product['price'] <= price_range[1] and 
                    product['stars'] >= min_rating and 
                    product['reviews'] >= min_reviews):
                    
                    col_idx = idx % 3
                    with cols[col_idx]:
                        st.markdown("""
                            <div class="product-card">
                                <div class="product-image">
                                    <img src="{}" alt="product image">
                                </div>
                            </div>
                        """.format(product['imgUrl'] if pd.notna(product.get('imgUrl')) else ""), 
                        unsafe_allow_html=True)
                        
                        st.markdown(f"**{product['title'][:100]}...**")
                        st.write(f"💰 Price: £{product['price']:.2f}")
                        st.write(f"⭐ Rating: {product['stars']:.1f}/5 ({int(product['reviews']):,} reviews)")
                        st.write(f"📁 {product['categoryName']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if pd.notna(product.get('productURL')):
                                st.markdown(f"[See on Amazon]({product['productURL']})")
                        with col2:
                            if st.button("See details", key=f"prod_{idx}"):
                                st.session_state.selected_product = product
                                st.session_state.current_page = 'detail'
                                st.rerun()
        else:
            st.warning("No products found matching your criteria.")
            
    except Exception as e:
        st.error(f"Error loading products: {str(e)}")

# Product detail page
elif st.session_state.current_page == 'detail' and st.session_state.selected_product is not None:
    show_product_detail(st.session_state.selected_product)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ By Aurélie")