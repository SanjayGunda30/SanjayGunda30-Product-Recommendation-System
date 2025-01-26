import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Custom CSS to set the background image
page_bg_img = '''
<style>
body {
background-image: url("Screenshot 2025-01-26 021205.png");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


# Load your data
data = pd.read_csv('amazon.csv').drop_duplicates(subset=['product_name'])
index = pd.Series(data.index, index=data['product_name']).drop_duplicates()

# Load necessary files
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open('liner.pkl', 'rb') as f:
    liner = pickle.load(f)
with open('user_sim.pkl', 'rb') as f:
    user_sim = pickle.load(f)  

def get_recommendations(name, liner=liner, top_n=50):
    idx = index[name]
    liner_scores = list(enumerate(liner[idx]))
    # Check if x[1] is a scalar before indexing
    liner_scores = sorted(liner_scores, key=lambda x: x[1] if np.isscalar(x[1]) else x[1][0], reverse=True)
    liner_scores = liner_scores[1:top_n+1]
    product_indices = [i[0] for i in liner_scores]
    return data.iloc[product_indices]

def get_similar_products_by_name(product_name, data, user_sim, top_n=50):
    

    # Check if the product name exists in the data
    if product_name not in data['product_name'].values:
        print(f"Product '{product_name}' not found in the data.")
        return pd.DataFrame(columns=['product_id', 'product_name'])

    # Get the product ID for the given product name
    product_id = data[data['product_name'] == product_name]['product_id'].iloc[0]

    # Get the similarity scores for the given product ID
    similar_scores = user_sim[product_id].sort_values(ascending=False)

    # Get the top N similar product IDs
    top_similar_ids = similar_scores.index[1:top_n + 1]

    # Get the product names for the top similar product IDs
    similar_products = data[data['product_id'].isin(top_similar_ids)][
        [ 'product_name']
    ]

    return similar_products

def hybrid_recommendation(product_name, data, user_sim, liner, top_n=5):
    
    # Get content-based recommendations
    contain_recomadation = get_recommendations(product_name, liner=liner, top_n=top_n)

    # Get collaborative filtering recommendations
    collab_recomadation = get_similar_products_by_name(product_name, data, user_sim, top_n=top_n)

    # Combine and deduplicate recommendations
    hybrid_recommendation = pd.concat([contain_recomadation, collab_recomadation])
    hybrid_recommendation = hybrid_recommendation.drop_duplicates()

    # Return the recommended product names
    return hybrid_recommendation





st.title('Product Recommendation System')
spro = st.selectbox('Select Product:', data['product_name'].values)
product_name = spro
if st.button('Recommend'):
    recommendations = hybrid_recommendation(product_name, data, user_sim, liner)
   #st.write("Recommendations:")
    for _, row in recommendations.iterrows():
        
        if 'product_link' in row and pd.notna(row['product_link']):
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin:5 px 0;">
                <h3>{row['product_name']}</h3>
                <a href="{row['product_link']}" target="_blank">Open Link</a>
                
            </div>
            """, unsafe_allow_html=True)


