import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- 0. PAGE CONFIG & THEME ---
st.set_page_config(page_title="The Wine Value Explorer", layout="wide", page_icon="🍷")

# Inject the overall "Wine Theme" for Streamlit UI elements
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp { background-color: #FCFBF9; }
    h1, h2, h3 { color: #600000 !important; font-family: 'Georgia', serif; }
    
    /* Custom Wine Buttons */
    div.stButton > button:first-child {
        background-color: #600000; color: white; border-radius: 20px;
        padding: 10px 25px; font-weight: bold; border: none;
    }
    div.stButton > button:hover { background-color: #800000; border: 1px solid #F7E7CE; color: #F7E7CE; }
    </style>
""", unsafe_allow_html=True)


# --- 1. HELPER FUNCTIONS ---

def apply_card_styling(html_content, height_px="600"):
    """Wraps raw Plotly HTML in a styled CSS 'card' with a faded bottom edge."""
    styled_head = f"""
        <style>
            body {{ margin: 0; padding: 0; background: transparent; }}
            .plotly-graph-div, .js-plotly-plot, .plot-container {{ width: 100% !important; }}
            .plotly-graph-div {{ height: {height_px}px !important; }}
            .card-wrapper {{
                position: relative;
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 24px;
                padding: 18px 18px 8px 18px;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.10);
            }}
            .card-wrapper::after {{
                content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 60px;
                background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%);
                border-radius: 0 0 24px 24px; pointer-events: none;
            }}
        </style>
        </head>
    """
    html_content = html_content.replace("</head>", styled_head)
    html_content = html_content.replace("<body>", '<body style="margin:0; padding:0;">\n<div class="card-wrapper">', 1)
    html_content = html_content.replace("</body>", "</div>\n</body>", 1)
    return html_content

def load_and_render_html(filename, height, scroll=False):
    """Safely loads an HTML file. Prevents app crashes if file is missing."""
    try:
        html_path = Path(__file__).with_name(filename)
        raw_html = html_path.read_text(encoding="utf-8")
        styled_html = apply_card_styling(raw_html, height_px=str(height))
        # We add a little buffer to the component height so the shadow isn't cut off
        components.html(styled_html, height=height + 40, scrolling=scroll)
    except FileNotFoundError:
        st.error(f"Visualization file missing: `{filename}`. Please ensure it is in the same directory.")


# --- 2. CACHING THE HEAVY LIFTERS ---

@st.cache_data 
def load_wine_data():
    return pd.read_parquet("sommelier_data.parquet")

@st.cache_resource 
def load_nlp_models():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        matrix = scipy.sparse.load_npz("tfidf_matrix.npz")
        return vectorizer, matrix
    except FileNotFoundError:
        return None, None

df_sommelier = load_wine_data()
tfidf_vectorizer, tfidf_matrix = load_nlp_models()


# --- 3. NLP SEARCH FUNCTIONS ---

def plot_sommelier_results(top_wines):
    categories = ['Value Index', 'Review Score', 'Price Efficiency', 'Flavor Match', 'Relative Popularity']
    # Wine-themed colors for the radar chart (Burgundy, Gold, Rose)
    colors = ['#600000', '#D4AF37', '#C08081'] 

    fig = go.Figure()

    for count, (i, row) in enumerate(top_wines.iterrows()):
        color = colors[count % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=[row['tvi_scaled'], row['points_scaled'], row['price_efficiency'], 
               row['flavor_match_normalized'], row['popularity_scaled']],
            theta=categories,
            fill='toself',
            name=row['title'][:30] + "...",
            line=dict(color=color, width=3),
            fillcolor=color,
            marker=dict(color=color),
            opacity=0.7
        ))

    fig.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
      showlegend=True,
      title="<b>Why the AI Picked These</b><br><sup>Comparing flavor match and statistical value</sup>",
      margin=dict(t=80, b=40, l=40, r=40)
    )
    return fig

def get_concierge_recommendation(user_text, max_price=50):
    user_vec = tfidf_vectorizer.transform([user_text])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    sim_scores_df = df_sommelier.copy()
    sim_scores_df['similarity_score'] = cosine_sim
    
    affordable = sim_scores_df[sim_scores_df['price'] <= max_price]
    if affordable.empty:
        return pd.DataFrame() # Return empty if no wines fit the budget
        
    top_matches = affordable.nlargest(15, 'similarity_score')
    winners = top_matches.sort_values(by='True Value Index', ascending=False).head(3).copy()

    max_sim = top_matches['similarity_score'].max()
    winners['flavor_match_normalized'] = winners['similarity_score'] / max_sim if max_sim > 0 else 0

    scaler = MinMaxScaler()
    try:
        winners['tvi_scaled'] = scaler.fit_transform(winners[['True Value Index']])
        winners['points_scaled'] = scaler.fit_transform(winners[['points']])
    except Exception:
        winners['tvi_scaled'] = 1.0
        winners['points_scaled'] = 1.0
    
    winners['price_efficiency'] = 1 - (winners['price'] / max_price)
    winners['popularity_scaled'] = winners['similarity_score'] / winners['similarity_score'].max() if winners['similarity_score'].max() > 0 else 0

    return winners


# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("SDAV Final Project")
    st.markdown("---")
    st.write("[📓 Explainer Notebook](https://github.com/a-thansen/SDAV/blob/main/Explainer_Notebook.ipynb)")
    st.write("[💻 GitHub Repository](https://github.com/a-thansen/SDAV)")
    st.write("[📊 Kaggle Dataset](https://www.kaggle.com/datasets/manyregression/updated-wine-enthusiast-review)")


# --- 5. NARRATIVE UI ---

st.title("🍷 The Wine Value Explorer")
st.subheader("Uncovering the True Value in the Global Wine Market")
st.markdown("""
Let’s be honest: walking into a wine shop is stressful. You’re standing there looking at a wall of labels, trying to guess if that \$40 bottle is actually four times better than the \$10 one next to it. Usually, we just pick the prettiest label or the most expensive thing we can afford and hope for the best.

We wanted to fix that. We took 76,000 professional reviews—basically, we let the world’s best sommeliers do the homework for us—and turned their expertise into a giant database. We’re looking past the fancy marketing and the "prestige" names to find the bottles that punch way above their weight class. This isn't about being a snob; it’s about making sure you never get ripped off again.""")
st.divider()

# --- SECTION 1: THE ILLUSION OF PRICE ---
st.header("1. The Illusion of Price")

col1, col2 = st.columns([1, 2])

with col1:
    # Main narrative text
    st.markdown("""
    Ever had a \$50 bottle that tasted... just okay? It’s not your palate; it’s the market. 
    When we look at the data for 76,000 wines, we see a price cloud. There are 
    plenty of \$20 bottles that experts rate higher than \$200 ones.

    The truth is, after a certain point, you aren't paying for better grapes—you’re 
    paying for the prestige. You’re paying for the history of a French chateau or 
    a famous California zip code. There is clearly a price ceiling. Once you hit it, 
    the quality stops going up, but the price keeps climbing. 
    
    Our goal is to stay right under that ceiling where the real deals live.
    """)

    # Refactored Data Details & Bias Note
    st.info("""
    **The Data Behind the Story**
    * **Source:** 76,000+ cleaned professional reviews from *Wine Enthusiast*.
    * **Metrics:** We analyzed the relationship between the 100-point scale and market pricing.
    * **Fairness Note:** Because our source is US-based, roughly 45% of the entries are American wines. To ensure a small winery in Portugal or Chile gets a fair shake against a California giant, we use Z-score normalization (see more in the next section) to rank quality relative to each region's own average.
    """)

with col2:
    # Render the interactive price ceiling plot
    load_and_render_html("wine_price_ceiling.html", height=600)

# --- SECTION 2: MARKET DNA ---
st.header("2. Understanding the Market: Finding the Neighborhoods")

# Create two columns to balance the text and the tool logic
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    To make sense of the mess, we used a machine learning technique called **K-Means Clustering**. 
    Think of it as asking the computer to find the natural "neighborhoods" of the wine world based 
    only on price and quality. 

    **The four AI-identified groups:**
    * **🛒 Budget Staple:** Great for the price; reliable everyday picks.
    * **💎 Hidden Gem:** High-quality wine at a surprisingly low price.
    * **✅ Solid Choice:** A step up in quality that justifies the cost.
    * **✨ Premium Choice:** Top-tier wine where you pay for excellence.
    """)

with col_right:
    st.markdown("### The TVI Compass")
    # Placing the boxes and expander in the same column balances the height
    st.error("**Negative Score:** You’re paying a 'Prestige Tax' for the label name.")
    st.success("**Score of +2.0 or higher:** 'Hidden Gem' DNA—statistically superior.")
    
    with st.expander("🤓 How is 'Value' calculated?"):
        st.markdown("""
        We use **Z-Scores** to compare a \$15 bottle to a \$500 bottle fairly:
        1. **Standardize Quality:** Steps above/below average score.
        2. **Standardize Price:** Steps above/below average price.
        3. **The Result:** $TVI = Z_{points} - Z_{price}$
        If a wine's quality is way higher than its price suggests, its TVI shoots up. This allows us to find the overachievers in any category, from grocery store reds to elite sparkling wines.
        """)

# Render the plot full-width below the balanced text section
st.markdown("""---
**🔍 Visual Guide:** Larger circles in the plot below represent a higher TVI.""")
load_and_render_html("wine_tribes_clusters_emojis.html", height=760)

# --- SECTION 3: THE LANDSCAPE ---
st.header("3. The Landscape")

# A short intro before the dynamic split
st.markdown("""
If you just follow the crowd, you’ll usually end up in places like Napa Valley or Bordeaux. 
They make incredible wine, but they also charge a "prestige tax" just for the privilege of 
knowing their names. 

To find the true outliers, we have to look past the household names. We filtered our 
entire database using our **True Value Index (TVI)** to look only at wines that sit in 
the top tiers of market efficiency. What you are seeing below is a 
Map of Bargains.
""")

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Volume"

view_cols = st.columns([8, 2])

if st.session_state.view_mode == "Volume":
    with view_cols[0]: 
        st.markdown("### 📦 Currently Viewing: Market Volume")
    with view_cols[1]:
        if st.button("🔄 Switch to Value Titans"):
            st.session_state.view_mode = "Quality"
            st.rerun()
            
    # The prompt to click the button
    st.info("""
    **This is the market as you usually see it.** These are the Market Giants that take up 
    all the shelf space at your local store. But what happens if we stop measuring by how *much* wine they make, and start measuring by the **value** they deliver? 
    
    **Hit the button above to redraw the map.**
    """)
    load_and_render_html("value_volume_treemap.html", height=760)

else:
    with view_cols[0]: 
        st.markdown("### 💎 Currently Viewing: Weighted Quality (TVI)")
    with view_cols[1]:
        if st.button("🔄 Switch to Market Giants"):
            st.session_state.view_mode = "Volume"
            st.rerun()
            
    # The "Aha!" moment text
    st.success("""
    **Notice how the world just flipped?** When we redraw the map based on our True Value Index, 
    the giants shrink. Suddenly, countries like Portugal, Chile, and Southern Italy become the main characters. 
    
    These are the Value Titans. They have the right soil and centuries of tradition, but they haven't 
    spent billions on marketing yet. This map is your cheat sheet: look here if you want top shelf 
    taste on a lower budget.
    """)
    load_and_render_html("weighted_quality_treemap.html", height=760)

st.header("4. The Palette: Flavor DNA")

st.markdown("""
Knowing which countries offer the best value is a great start, but you cannot drink geography. To find a wine you will actually love, we have to talk about flavor. 

Most wine descriptions are full of weird words like forest floor or wet stones, so we used AI to cut through the jargon and find the actual **Flavor Fingerprint** of each grape. Whether you prefer your wine **Oaky and Toasty** or **Citrus and Crisp**, the matrix below shows you exactly which grapes give you the most bang for your buck in that specific category. 

Think of it as a shortcut to finding your new favorite variety based on what you actually like to taste, rather than what an expert says you should like. You now know which grapes offer the best value, **but how do you know which of those grapes or flavors will best complement your next meal?**
""")

load_and_render_html("flavor_value_matrix_landscape.html", height=500)

st.divider()

# Section 5
st.header("5. The AI Sommelier")
st.markdown("""
This is where the math meets the menu. Our AI Sommelier takes your dinner plans and your favorite flavor profiles to find the exact bottles that hit the high-value sweet spot. 

Instead of guessing in the wine aisle, tell us what you are eating. We will scan our 76,000-bottle database to find the three **Hidden Gems** that will make your meal—and your wallet—happy.
""")
col_food, col_flavor = st.columns(2)
with col_food:
    food_input = st.text_input("🍽️ What's on the menu?", placeholder="e.g. Grilled Salmon, Spicy Tacos")
    max_budget = st.slider("Max Price ($)", min_value=10, max_value=175, value=30, step=5)

with col_flavor:
    flavor_pref = st.multiselect("🍇 Preferred wine style...", 
                                 ["Oaky/Toasty", "Dark Fruit", "Citrus/Acidic", "Earthy/Savory", "Tropical/Sweet"])

if st.button("Generate My Personalized Value Pairing"):
    if food_input or flavor_pref:
        with st.spinner('Airing the data and checking the cellar...'):
            combined_query = f"{food_input} {' '.join(flavor_pref)}"
            results = get_concierge_recommendation(combined_query, max_price=max_budget)
            
            if results.empty:
                st.warning(f"We couldn't find any high-match wines under ${max_budget}. Try increasing your budget!")
            else:
                st.success("The Sommelier has found bottles that match your palate perfectly. Double-click the descriptions to see the full tasting notes and why these wines are such great values.")
                
                st.write("### The Sommelier's Selection")
                
                # Update the columns we show to the user
                display_cols = ['title', 'variety', 'price', 'points', 'True Value Index', 'description']
                
                # Render the table WITHOUT the index number
                st.dataframe(
                    results[display_cols].style.format({
                        'price': '${:.2f}', 
                        'True Value Index': '{:.2f}'
                    }), 
                    use_container_width=True,
                    hide_index=True  # <--- THIS REMOVES THE INDEX NUMBER
                )
                
                # Render the Radar Chart
                radar_fig = plot_sommelier_results(results)
                st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.warning("Please tell us what you are eating or select a flavor profile first!")
