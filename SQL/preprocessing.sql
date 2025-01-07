-- 1. Clean Ratings View
-- Filter valid reviews (stars > 0 and <= 5, reviews > 0)
CREATE OR REPLACE VIEW `bot-recommendation.amazonuk_data.CleanRatings` AS
SELECT DISTINCT
    asin,
    stars,
    reviews,
    LOG(reviews + 1) as reviews_log,
    CASE 
        WHEN stars >= 4 THEN 1
        ELSE 0
    END as is_high_rated
FROM `bot-recommendation.amazonuk_data.Ratings`
WHERE stars > 0 
AND stars <= 5 
AND reviews > 0;

-- 2. Clean Products View with Price Features
-- Filter valid prices and add price transformations
CREATE OR REPLACE VIEW `bot-recommendation.amazonuk_data.CleanProducts` AS
WITH price_percentile AS (
    SELECT 
        price,
        PERCENTILE_CONT(price, 0.99) OVER() as price_99th_percentile
    FROM `bot-recommendation.amazonuk_data.Products`
    WHERE price > 0
),
price_stats AS (
    SELECT 
        category_id,
        AVG(price) as avg_category_price
    FROM `bot-recommendation.amazonuk_data.Products`
    WHERE price > 0
    GROUP BY category_id
)
SELECT DISTINCT
    p.asin,
    p.title,
    p.img_url,
    p.product_url,
    p.price,
    p.is_best_seller,
    p.category_id,
    LN(p.price + 1) as price_log,
    p.price / ps.avg_category_price as price_ratio_to_category,
    CASE 
        WHEN p.price <= ps.avg_category_price * 0.4 THEN 'very_cheap'
        WHEN p.price <= ps.avg_category_price * 0.8 THEN 'cheap'
        WHEN p.price <= ps.avg_category_price * 1.2 THEN 'medium'
        WHEN p.price <= ps.avg_category_price * 1.6 THEN 'expensive'
        ELSE 'very_expensive'
    END as price_category
FROM `bot-recommendation.amazonuk_data.Products` p
JOIN price_stats ps ON p.category_id = ps.category_id
CROSS JOIN (SELECT price_99th_percentile FROM price_percentile LIMIT 1) pp
WHERE p.price > 0 
    AND p.price <= pp.price_99th_percentile;

-- 3. Feature Engineering View
-- Combines the previous views and adds calculated features
CREATE OR REPLACE VIEW `bot-recommendation.amazonuk_data.ProductFeatures` AS
WITH max_reviews AS (
    SELECT MAX(reviews) as max_rev
    FROM `bot-recommendation.amazonuk_data.CleanRatings`
),
normalized_scores AS (
    SELECT 
        p.*,
        COALESCE(r.stars, 0) as stars,
        COALESCE(r.reviews, 0) as reviews,
        COALESCE(r.reviews_log, 0) as reviews_log,
        COALESCE(r.is_high_rated, 0) as is_high_rated,
        COALESCE(s.bought_in_last_month, 0) as bought_in_last_month,
        COALESCE(
            -- Popularity score (comme dans preprocessing.py)
            (0.7 * (r.reviews_log / NULLIF(LOG(mr.max_rev + 1), 0)) + 0.3 * (r.stars / 5)),
            0
        ) as popularity_score,
        COALESCE(
            -- Value for money comme dans preprocessing.py
            (r.stars / NULLIF(p.price_log + 1, 0)) * (1 + r.reviews_log / NULLIF(LOG(mr.max_rev + 1), 0)),
            0
        ) as value_for_money,
        -- Price segment (NTILE comme qcut dans pandas)
        NTILE(10) OVER (ORDER BY p.price_log) as price_segment,
        -- Review segment
        CASE 
            WHEN NTILE(5) OVER (ORDER BY COALESCE(r.reviews_log, 0)) = 1 THEN 'very_low'
            WHEN NTILE(5) OVER (ORDER BY COALESCE(r.reviews_log, 0)) = 2 THEN 'low'
            WHEN NTILE(5) OVER (ORDER BY COALESCE(r.reviews_log, 0)) = 3 THEN 'medium'
            WHEN NTILE(5) OVER (ORDER BY COALESCE(r.reviews_log, 0)) = 4 THEN 'high'
            ELSE 'very_high'
        END as review_segment
    FROM `bot-recommendation.amazonuk_data.CleanProducts` p
    LEFT JOIN `bot-recommendation.amazonuk_data.CleanRatings` r 
        ON p.asin = r.asin
    LEFT JOIN `bot-recommendation.amazonuk_data.Sales` s 
        ON p.asin = s.asin
    CROSS JOIN max_reviews mr
)
SELECT DISTINCT * FROM normalized_scores;


-- 4. Analytics View
-- View for descriptive statistics
CREATE OR REPLACE VIEW `bot-recommendation.amazonuk_data.ProductAnalytics` AS
SELECT 
    category_id,
    COUNT(DISTINCT asin) as product_count,
    AVG(popularity_score) as avg_popularity,
    AVG(value_for_money) as avg_value_for_money,
    AVG(stars) as avg_stars,
    AVG(price) as avg_price,
    AVG(reviews) as avg_reviews,
    COUNT(CASE WHEN is_best_seller THEN 1 END) as bestseller_count
FROM `bot-recommendation.amazonuk_data.ProductFeatures`
GROUP BY category_id;