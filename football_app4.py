import pandas as pd
import streamlit as st
import re
import requests
import plotly.graph_objects as go

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Football Player Similarity Finder For Scouting and Analysis", layout="wide")
st.title("⚽ Football Player Similarity Finder For Scouting and Analysis")

st.markdown("""
This machine learning scouting tool identifies football players with **similar statistical profiles**
using **cosine similarity**.  

Select a player to discover the **Top 5 most similar players** based on performance statistics.
""")

# --------------------------------
# Tabs
# --------------------------------
tab1, tab2 = st.tabs(["🔎 Similarity Finder", "📊 Stat Indicators"])

# ==================================
# TAB 1 — SIMILARITY APP
# ==================================
with tab1:

    # -----------------------------
    # 1 Load Dataset
    # -----------------------------
    @st.cache_data
    def load_data():
        data = pd.read_csv("players_2526.csv")
        return data

    data = load_data()

    # -----------------------------
    # 2 Preprocessing
    # -----------------------------
    data = data.drop("Rk", axis=1, errors="ignore")

    data["Nation"] = data["Nation"].apply(lambda x: "".join(re.findall(r"[A-Z]", str(x))))

    league_patterns = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    regex_pattern = "|".join(map(re.escape, league_patterns))
    data["Comp"] = data["Comp"].str.extract(f"({regex_pattern})", expand=False)

    data["Pos"] = data["Pos"].apply(lambda x: str(x).split(",")[0])

    data["Age"] = data["Age"].fillna(data["Age"].median()).astype("Int64")
    data["Born"] = data["Born"].fillna(data["Born"].median()).astype("Int64")

    data.dropna(subset=["SoT%", "G/Sh", "G/SoT"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # -----------------------------
    # Feature Selection
    # -----------------------------
    numerical_data = data.select_dtypes(include=["number"]).copy()

    selector = VarianceThreshold(threshold=0.01)
    selected_features = selector.fit_transform(numerical_data)
    selected_columns = numerical_data.columns[selector.get_support()]

    data_selected = pd.DataFrame(selected_features, columns=selected_columns)

    # -----------------------------
    # Feature Scaling
    # -----------------------------
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_selected)

    # -----------------------------
    # Similarity Matrix
    # -----------------------------
    similarity_matrix = cosine_similarity(scaled_features)

    # -----------------------------
    # Player Image API
    # -----------------------------
    @st.cache_data
    def get_player_image(player_name):
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={player_name}"
        try:
            r = requests.get(url).json()
            if r["player"]:
                return r["player"][0]["strThumb"]
        except:
            return None
        return None

    # -----------------------------
    # Recommendation Function
    # -----------------------------
    def hybrid_recommend(player_name, top_n=5):
        if player_name not in data["Player"].values:
            return None
        idx = data[data["Player"] == player_name].index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:top_n + 1]
        player_indices = [i[0] for i in scores]
        return data.iloc[player_indices]

    # -----------------------------
    # Radar Chart
    # -----------------------------
    radar_stats = ["Gls", "Ast", "xG", "PrgP", "KP", "Tkl", "Int", "Touches"]

    def create_radar_chart(player1, player2):
        stats = [col for col in radar_stats if col in data.columns]
        try:
            p1 = data[data["Player"] == player1][stats].iloc[0].fillna(0).astype(float)
            p2 = data[data["Player"] == player2][stats].iloc[0].fillna(0).astype(float)
        except:
            return None

        # Normalize for better radar chart visualization
        if p1.max() > 0:
            p1 = p1 / p1.max()
        if p2.max() > 0:
            p2 = p2 / p2.max()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=p1.values,
            theta=stats,
            fill='toself',
            name=player1
        ))
        fig.add_trace(go.Scatterpolar(
            r=p2.values,
            theta=stats,
            fill='toself',
            name=player2
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"{player1} vs {player2} Radar Chart"
        )
        return fig

    # -----------------------------
    # Stats columns
    # -----------------------------
    stats_columns = data.select_dtypes(include=["number"]).columns.tolist()

    # -----------------------------
    # UI
    # -----------------------------
    player_input = st.selectbox("Select Player", sorted(data["Player"].unique()))

    if st.button("Find Similar Players"):
        results = hybrid_recommend(player_input)

        if results is None:
            st.error("Player not found.")
        else:
            st.subheader("Top 5 Similar Players")
            for _, row in results.iterrows():
                img = get_player_image(row["Player"])
                col1, col2 = st.columns([3, 4])

                with col1:
                    if img:
                        st.image(img, use_container_width=True)

                with col2:
                    st.markdown(f"### {row['Player']}")
                    st.write(f"**Age:** {row['Age']}")
                    st.write(f"**Position:** {row['Pos']}")
                    st.write(f"**Team:** {row['Squad']}")
                    st.write(f"**League:** {row['Comp']}")
                    st.write(f"**Nationality:** {row['Nation']}")

                    # -------------------------
                    # View All Stats
                    # -------------------------
                    with st.expander("📊 View All Player Stats"):
                        stats_df = pd.DataFrame(row[stats_columns]).reset_index()
                        stats_df.columns = ["Stat", "Value"]
                        st.dataframe(stats_df, use_container_width=True)

                    # -------------------------
                    # Radar Chart
                    # -------------------------
                    with st.expander("📈 Compare Radar Chart"):
                        radar_fig = create_radar_chart(player_input, row["Player"])
                        if radar_fig is not None:
                            st.plotly_chart(radar_fig, use_container_width=True)
                        else:
                            st.warning("Radar chart cannot be generated for this player.")

                st.divider()

# ==================================
# TAB 2 — STAT GLOSSARY
# ==================================
with tab2:
    st.header("Football Statistics Indicators")

    st.subheader("Playing Time & Appearances")
    st.markdown("""
MP – Matches played  
Starts – Games started  
Min – Minutes played  
90s – Number of full 90-minute matches played
""")

    st.subheader("Attacking Stats")
    st.markdown("""
Gls – Goals scored  
Ast – Assists provided  
G+A – Goals + Assists  
xG – Expected goals  
xAG – Expected assists  
npxG – Non-penalty expected goals  
G-PK – Goals excluding penalties
""")

    st.subheader("Defensive Stats")
    st.markdown("""
Tkl – Total tackles  
TklW – Tackles won  
Blocks – Blocks made  
Int – Interceptions  
Tkl+Int – Combined tackles and interceptions  
Clr – Clearances  
Err – Errors leading to goals
""")

    st.subheader("Passing & Creativity")
    st.markdown("""
PrgP – Progressive passes  
PrgC – Progressive carries  
KP – Key passes  
Cmp% – Pass completion percentage  
xA – Expected assists  
PPA – Passes into penalty area
""")

    st.subheader("Goalkeeping Stats")
    st.markdown("""
GA – Goals conceded  
Saves – Saves made  
Save% – Save percentage  
CS – Clean sheets  
CS% – Clean sheet percentage  
PKA – Penalties faced  
PKsv – Penalty saves
""")

    st.subheader("Possession & Ball Control")
    st.markdown("""
Touches – Total touches  
Carries – Ball carries  
PrgR – Progressive runs  
Mis – Miscontrols  
Dis – Times dispossessed
""")

    st.subheader("Miscellaneous")
    st.markdown("""
CrdY – Yellow cards  
CrdR – Red cards  
PKwon – Penalties won  
PKcon – Penalties conceded  
Recov – Ball recoveries
""")