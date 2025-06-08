import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="⋆. 𐙚 ˚ liberai ˊˎ-", layout="centered")

main_bg = "display.gif"
main_bg_ext = "gif"

with open(main_bg, "rb") as file:
    base64_gif = base64.b64encode(file.read()).decode()

st.markdown(
    f"""
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/{main_bg_ext};base64,{base64_gif}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Load artifacts
@st.cache_resource(show_spinner=True)
def load_artifacts():
    artifacts_dir = "."
    try:
        with open(os.path.join(artifacts_dir, 'books_df.pkl'), 'rb') as f:
            books_df = pickle.load(f)
        with open(os.path.join(artifacts_dir, 'cosine_sim.pkl'), 'rb') as f:
            cosine_sim = pickle.load(f)
        with open(os.path.join(artifacts_dir, 'kmeans.pkl'), 'rb') as f:
            kmeans = pickle.load(f)
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run the main.ipynb script first.")
        return None, None, None
    return books_df, cosine_sim, kmeans

books_df, cosine_sim, kmeans = load_artifacts()

# Main recommendation
def recommend_books_hybrid(selected_titles, books_df, cosine_sim_matrix, top_n):
    # Create a mapping from title to index for quick lookup
    title_to_index = pd.Series(books_df.index, index=books_df['title'].str.lower()).to_dict()
    
    indices = [title_to_index.get(title.lower()) for title in selected_titles if title.lower() in title_to_index]

    if not indices:
        return pd.DataFrame()

    # Calculate the average similarity score for the input books
    sim_scores = np.mean(cosine_sim_matrix[indices], axis=0)
    
    # Get clusters of selected books
    selected_clusters = books_df.iloc[indices]['cluster'].unique()
    candidate_mask = books_df['cluster'].isin(selected_clusters)
    candidate_indices = np.where(candidate_mask)[0]
    candidate_indices = [i for i in candidate_indices if i not in indices]
    
    # Create a DataFrame of candidate books
    recommendations = pd.DataFrame({
        'index': candidate_indices,
        'similarity_score': sim_scores[candidate_indices]
    })
    
    # Merge with original dataframe to get metadata for hybrid scoring
    recommendations = recommendations.merge(books_df[['rating', 'numRatings','description']], left_on='index', right_index=True)
    
    # Calculate hybrid score
    recommendations['popularity_score'] = np.log1p(recommendations['numRatings'])
    recommendations['popularity_score'] = recommendations['popularity_score'] / recommendations['popularity_score'].max()
    recommendations['rating_score'] = recommendations['rating'] / 5.0
    recommendations['hybrid_score'] = (
        (0.5 * recommendations['similarity_score']) +
        (0.3 * recommendations['rating_score']) +
        (0.2 * recommendations['popularity_score'])
    )
    
    # Sort and get top N
    top_recommendations = recommendations.sort_values(by='hybrid_score', ascending=False).head(top_n)
    
    # Get final book details
    final_indices = top_recommendations['index']
    final_recs = books_df.iloc[final_indices][['title', 'author', 'genres', 'rating', 'numRatings','description']].copy()
    final_recs['Hybrid Score'] = np.round(top_recommendations['hybrid_score'], 3)
    
    return final_recs

# UI
if books_df is not None:
    st.title('_⋆. 𐙚 ˚ liberai ˊˎ-_')
    st.markdown('╰┈➤ find a recommended book based on your favourites with **liberai** ! ── .✦')
    top_n = st.number_input('how many books do u want to see? [10-50] 👀',min_value=10, max_value=50, value=10, step=1)

    # User input
    selected_books = st.multiselect(
        "tell **liberai** your top 3 books! 📚",
        placeholder="select here",
        options=books_df['title'].tolist(),
        max_selections=3
    )

    # Button
    if st.button("find ur books!", type="primary"):
        if len(selected_books) == 3:
            with st.spinner("finding great books for you..."):
                recommendations = recommend_books_hybrid(selected_books, books_df, cosine_sim,top_n=top_n)
                if not recommendations.empty:
                    st.success("thanks for waiting!")
                    st.subheader(f"ᯓ★ here are your top {len(recommendations)} recommendations:")
                    st.markdown("---")

                    for _, row in recommendations.iterrows():
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"by *{row['author'].replace('_', ' ').title()}*")
                        st.caption(f"Genres: {row['genres'].lower()}")
                        description = row.get('description', '')
                        max_chars = 500
                        if pd.notna(description) and description.strip():
                            short_desc = (description[:max_chars] + '...') if len(description) > max_chars else description
                            st.markdown(f"{short_desc.strip()}")
                        else:
                            st.markdown("no description available :(")
                        st.caption(f"Rating: {row['rating']}")
                        st.markdown("---")
                else:
                    st.warning("oh no! could not generate recommendations :( please try different books!")
        else:
            st.warning("oh no! that's not the desired amount :(")
    
    # Discover books using user-inputted genre from clusters
    st.markdown('╰┈➤ explore ur genres! ── .✦')
    
    # User input
    selected_cluster_name = st.selectbox(
        "choose a genre! 📚",
        sorted(books_df['cluster_name'].unique()),
        index=0,
        placeholder="select here"
    )
    
    # Button for genre cluster
    if st.button("find ur books!", type="primary", key="genre_button"):
        filtered_books = books_df[books_df['cluster_name'] == selected_cluster_name]
        st.success("thanks for waiting!")
        st.subheader(f"ᯓ★ showing {len(filtered_books)} books in _{selected_cluster_name}_")
        for _, row in filtered_books.iterrows():
            st.markdown(f"**{row['title']}**")
            st.caption(f"by *{row['author'].replace('_', ' ').title()}*")
            st.caption(f"Genres: {row['genres'].lower()}")
            description = row.get('description', '')
            max_chars = 500
            if pd.notna(description) and description.strip():
                short_desc = (description[:max_chars] + '...') if len(description) > max_chars else description
                st.markdown(f"{short_desc.strip()}")
            else:
                st.markdown("no description available :(")
            st.caption(f"Rating: {row['rating']}")
            st.markdown("---")

else:
    st.stop()