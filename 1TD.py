import streamlit as st
import os
import pandas as pd
import numpy as np
import logging
import datetime
import time

# --- ML & Plotting Imports ---
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go

# --- Basic App Configuration ---
st.set_page_config(
    page_title="NFL 1st TD Scorer Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.options.mode.chained_assignment = None

TODAY = datetime.date.today()
SEASONS = [2024, 2025]
CURRENT_SEASON = TODAY.year if TODAY.month > 6 else TODAY.year - 1
EWMA_SPAN = 4

# --- File Paths ---
# These files MUST be in your GitHub repo now
DATA_DIR = "nfl_data_cache"
MODEL_PATH = os.path.join(DATA_DIR, "td_predictor_model.json")
PBP_CACHE_PATH = os.path.join(DATA_DIR, "pbp_data.parquet")
SCHEDULE_CACHE_PATH = os.path.join(DATA_DIR, "schedule_data.parquet")
ROSTER_CACHE_PATH = os.path.join(DATA_DIR, "rosters_data.parquet")
DEPTH_CHART_CACHE_PATH = os.path.join(DATA_DIR, "depth_charts.parquet")

# --- Simplified Data Loading ---
@st.cache_data
def load_all_data():
    """Loads all necessary data from the parquet files in the repo."""
    try:
        data = {
            "pbp": pd.read_parquet(PBP_CACHE_PATH),
            "schedule": pd.read_parquet(SCHEDULE_CACHE_PATH),
            "rosters": pd.read_parquet(ROSTER_CACHE_PATH),
            "depth_charts": pd.read_parquet(DEPTH_CHART_CACHE_PATH)
        }
        return data
    except Exception as e:
        st.error(f"Failed to read data files from the repository. Make sure the .parquet files are there. Error: {e}")
        return None

# --- Feature Engineering & Prediction Logic (This code does not need to change) ---
class FeatureEngineeringEngine:
    # ... (paste the entire FeatureEngineeringEngine class here, unchanged)
    def __init__(self, all_data):
        self.raw_data = all_data
        self.features_df = None

    def run(self):
        self._create_base_player_games_from_pbp()
        self._create_target_variable()
        player_usage = self._calculate_player_usage()
        defense_vuln = self._calculate_defense_vulnerabilities()
        depth_charts = self._process_depth_charts()
        self._merge_features(player_usage, defense_vuln, depth_charts)
        return self.features_df.dropna(subset=['ewma_opp_share', 'ewma_rz_share'])

    def _create_base_player_games_from_pbp(self):
        pbp = self.raw_data['pbp']
        id_cols = ['rusher_player_id', 'receiver_player_id', 'passer_player_id']
        player_games = pd.melt(pbp, id_vars=['game_id', 'season', 'week', 'posteam'], value_vars=id_cols, value_name='gsis_id').dropna(subset=['gsis_id'])
        base_df = player_games[['game_id', 'season', 'week', 'posteam', 'gsis_id']].drop_duplicates()
        base_df.rename(columns={'posteam': 'team'}, inplace=True)
        roster_info = self.raw_data['rosters'][['gsis_id', 'full_name', 'position']].drop_duplicates(subset=['gsis_id'])
        self.features_df = pd.merge(base_df, roster_info, on='gsis_id', how='left')
        self.features_df = self.features_df[self.features_df['position'].isin(['RB', 'WR', 'TE', 'QB'])]

    def _create_target_variable(self):
        td_plays = self.raw_data['pbp'][(self.raw_data['pbp']['touchdown'] == 1) & (self.raw_data['pbp']['td_team'].notna())].copy()
        td_plays['scorer_id'] = td_plays['rusher_player_id'].combine_first(td_plays['receiver_player_id'])
        td_plays = td_plays.dropna(subset=['scorer_id'])
        first_td_scorers = td_plays.sort_values(by=['game_id', 'play_id']).groupby(['game_id', 'td_team']).first().reset_index()[['game_id', 'scorer_id']]
        first_td_scorers['is_first_td_scorer'] = 1
        self.features_df = pd.merge(self.features_df, first_td_scorers, left_on=['game_id', 'gsis_id'], right_on=['game_id', 'scorer_id'], how='left')
        self.features_df['is_first_td_scorer'].fillna(0, inplace=True)
        self.features_df.drop(columns=['scorer_id'], inplace=True)

    def _calculate_player_usage(self):
        pbp = self.raw_data['pbp']
        usage_plays = pbp[(pbp['play_type'].isin(['run', 'pass'])) & (pbp['rusher_player_id'].notna() | pbp['receiver_player_id'].notna())]
        usage_df = pd.melt(usage_plays, id_vars=['game_id', 'season', 'week', 'posteam', 'yardline_100'], value_vars=['rusher_player_id', 'receiver_player_id'], value_name='player_id').dropna()
        usage_df['is_touch'] = 1
        usage_df['is_rz_touch'] = np.where(usage_df['yardline_100'] <= 20, 1, 0)
        weekly_stats = usage_df.groupby(['player_id', 'game_id', 'season', 'week']).agg(touches=('is_touch', 'sum'), rz_touches=('is_rz_touch', 'sum')).reset_index()
        team_totals = weekly_stats.groupby('game_id').agg(team_touches=('touches', 'sum'), team_rz_touches=('rz_touches', 'sum')).reset_index()
        weekly_stats = weekly_stats.merge(team_totals, on='game_id')
        weekly_stats['opp_share'] = weekly_stats['touches'] / weekly_stats['team_touches']
        weekly_stats['rz_share'] = weekly_stats['rz_touches'] / weekly_stats['team_rz_touches']
        weekly_stats.sort_values(by=['player_id', 'season', 'week'], inplace=True)
        grouped = weekly_stats.groupby('player_id')
        weekly_stats['ewma_opp_share'] = grouped['opp_share'].transform(lambda x: x.ewm(span=EWMA_SPAN, adjust=False).mean().shift(1))
        weekly_stats['ewma_rz_share'] = grouped['rz_share'].transform(lambda x: x.ewm(span=EWMA_SPAN, adjust=False).mean().shift(1))
        return weekly_stats

    def _calculate_defense_vulnerabilities(self):
        pbp, rosters = self.raw_data['pbp'], self.raw_data['rosters']
        td_plays = pbp[(pbp['touchdown'] == 1) & (pbp['td_team'].notna())].copy()
        td_plays['scorer_id'] = td_plays['rusher_player_id'].combine_first(td_plays['receiver_player_id'])
        td_plays = td_plays.dropna(subset=['scorer_id'])
        player_info = rosters[['gsis_id', 'position']].drop_duplicates()
        td_plays = td_plays.merge(player_info, left_on='scorer_id', right_on='gsis_id', how='left')
        td_plays = td_plays[td_plays['position'].isin(['RB', 'WR', 'TE', 'QB'])]
        by_pos = td_plays.groupby(['defteam', 'position']).size().reset_index(name='count')
        total = td_plays.groupby('defteam').size().reset_index(name='total')
        vuln = by_pos.merge(total, on='defteam')
        vuln['def_vuln_rate'] = vuln['count'] / vuln['total']
        return vuln[['defteam', 'position', 'def_vuln_rate']]

    def _process_depth_charts(self):
        depth_charts = self.raw_data['depth_charts']
        depth_multipliers = {'RB1': 1.2, 'WR1': 1.15, 'TE1': 1.1, 'QB1': 1.05, 'RB2': 0.9, 'WR2': 1.0, 'TE2': 0.85}
        depth_charts['depth_multiplier'] = depth_charts['depth_team'].map(depth_multipliers).fillna(0.7)
        return depth_charts[['season', 'week', 'team', 'gsis_id', 'depth_multiplier']]

    def _merge_features(self, player_usage, defense_vuln, depth_charts):
        usage_to_merge = player_usage[['game_id', 'player_id', 'ewma_opp_share', 'ewma_rz_share']]
        self.features_df = pd.merge(self.features_df, usage_to_merge, left_on=['game_id', 'gsis_id'], right_on=['game_id', 'player_id'], how='left')
        self.features_df = pd.merge(self.features_df, depth_charts, on=['season', 'week', 'team', 'gsis_id'], how='left')
        self.features_df['depth_multiplier'].fillna(0.6, inplace=True)
        schedule = self.raw_data['schedule'][['game_id', 'home_team', 'away_team']]
        self.features_df = self.features_df.merge(schedule, on='game_id')
        self.features_df['opponent'] = np.where(self.features_df['team'] == self.features_df['home_team'], self.features_df['away_team'], self.features_df['home_team'])
        self.features_df = pd.merge(self.features_df, defense_vuln, left_on=['opponent', 'position'], right_on=['defteam', 'position'], how='left')
        self.features_df['def_vuln_rate'].fillna(self.features_df['def_vuln_rate'].mean(), inplace=True)

class PredictionPipeline:
    # ... (paste the entire PredictionPipeline class here, unchanged)
    def __init__(self, all_data, week, season):
        self.all_data = all_data
        self.week = week
        self.season = season
        self.player_usage = self._calculate_player_usage()
        self.defense_vuln = self._calculate_defense_vulnerabilities()
        self.depth_charts = self._process_depth_charts()

    def run(self):
        upcoming_games = self.all_data['schedule'][(self.all_data['schedule']['season'] == self.season) & (self.all_data['schedule']['week'] == self.week)]
        if upcoming_games.empty:
            st.warning(f"No upcoming games found for week {self.week}.")
            return None
        base_df = self._create_prediction_base(upcoming_games)
        latest_usage = self.player_usage.groupby('player_id').last().reset_index()[['player_id', 'ewma_opp_share', 'ewma_rz_share']]
        features = self._merge_prediction_features(base_df, latest_usage)
        return features.fillna(0)

    def _create_prediction_base(self, upcoming_games):
        rosters = self.all_data['rosters'][(self.all_data['rosters']['season'] == self.season) & (self.all_data['rosters']['position'].isin(['RB', 'WR', 'TE', 'QB']))]
        game_teams = pd.melt(upcoming_games, id_vars=['game_id'], value_vars=['home_team', 'away_team'], value_name='team')[['game_id', 'team']]
        prediction_base = pd.merge(rosters, game_teams, on='team')
        return prediction_base[['game_id', 'team', 'gsis_id', 'full_name', 'position']].drop_duplicates()

    def _merge_prediction_features(self, base_df, latest_usage):
        df = pd.merge(base_df, latest_usage, left_on='gsis_id', right_on='player_id', how='left')
        latest_depths = self.depth_charts.sort_values('week').groupby('gsis_id').last().reset_index()
        df = pd.merge(df, latest_depths[['gsis_id', 'depth_multiplier']], on='gsis_id', how='left')
        df['depth_multiplier'].fillna(0.6, inplace=True)
        schedule = self.all_data['schedule'][['game_id', 'home_team', 'away_team']]
        df = df.merge(schedule, on='game_id')
        df['opponent'] = np.where(df['team'] == df['home_team'], df['away_team'], df['home_team'])
        df = pd.merge(df, self.defense_vuln, left_on=['opponent', 'position'], right_on=['defteam', 'position'], how='left')
        df['def_vuln_rate'].fillna(df['def_vuln_rate'].mean(), inplace=True)
        return df
    
    def _calculate_player_usage(self):
        pbp = self.all_data['pbp']
        usage_plays = pbp[(pbp['play_type'].isin(['run', 'pass'])) & (pbp['rusher_player_id'].notna() | pbp['receiver_player_id'].notna())]
        usage_df = pd.melt(usage_plays, id_vars=['game_id', 'season', 'week', 'posteam', 'yardline_100'], value_vars=['rusher_player_id', 'receiver_player_id'], value_name='player_id').dropna()
        usage_df['is_touch'] = 1
        usage_df['is_rz_touch'] = np.where(usage_df['yardline_100'] <= 20, 1, 0)
        weekly_stats = usage_df.groupby(['player_id', 'game_id', 'season', 'week']).agg(touches=('is_touch', 'sum'), rz_touches=('is_rz_touch', 'sum')).reset_index()
        team_totals = weekly_stats.groupby('game_id').agg(team_touches=('touches', 'sum'), team_rz_touches=('rz_touches', 'sum')).reset_index()
        weekly_stats = weekly_stats.merge(team_totals, on='game_id')
        weekly_stats['opp_share'] = weekly_stats['touches'] / weekly_stats['team_touches']
        weekly_stats['rz_share'] = weekly_stats['rz_touches'] / weekly_stats['team_rz_touches']
        weekly_stats.sort_values(by=['player_id', 'season', 'week'], inplace=True)
        grouped = weekly_stats.groupby('player_id')
        weekly_stats['ewma_opp_share'] = grouped['opp_share'].transform(lambda x: x.ewm(span=EWMA_SPAN, adjust=False).mean())
        weekly_stats['ewma_rz_share'] = grouped['rz_share'].transform(lambda x: x.ewm(span=EWMA_SPAN, adjust=False).mean())
        return weekly_stats
    
    def _calculate_defense_vulnerabilities(self):
        return FeatureEngineeringEngine(self.all_data)._calculate_defense_vulnerabilities()
    
    def _process_depth_charts(self):
        return FeatureEngineeringEngine(self.all_data)._process_depth_charts()

# --- Model Training Function ---
def run_training_process():
    # ... (paste the entire run_training_process function here, unchanged)
    status_placeholder = st.empty()
    with st.spinner("Loading base data..."):
        all_data = load_all_data()
        if not all_data:
            status_placeholder.error("Data loading failed. Cannot proceed with training.")
            return

    status_placeholder.info("Data loaded. Starting feature engineering...")
    time.sleep(1) 
    with st.spinner("Engineering features from play-by-play data... This may take a moment."):
        engine = FeatureEngineeringEngine(all_data)
        features_df = engine.run()

    if features_df.empty:
        status_placeholder.error("Feature engineering resulted in an empty dataset. Check data sources.")
        return
    
    status_placeholder.info(f"Feature engineering complete. Found {len(features_df)} samples. Starting model training...")
    time.sleep(1)

    with st.spinner("Training XGBoost Classifier..."):
        target_col = 'is_first_td_scorer'
        feature_cols = ['ewma_opp_share', 'ewma_rz_share', 'depth_multiplier', 'def_vuln_rate']
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        
        model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, 
            n_estimators=150, learning_rate=0.1, max_depth=4, 
            scale_pos_weight=scale_pos_weight, random_state=42
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, zero_division=0, output_dict=True)
        model.save_model(MODEL_PATH)

    status_placeholder.success(f"Model training complete and saved to `{MODEL_PATH}`!")
    
    st.subheader("Model Evaluation Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3))
    st.info("You can now navigate to the 'Prediction Dashboard'.")

# --- UI Helper & Display Functions ---
def generate_justification(player_row):
    # ... (paste the entire generate_justification function here, unchanged)
    justifications = []
    rz_share = player_row['ewma_rz_share']
    if rz_share > 0.35: justifications.append(f"**Elite Red Zone Usage** ({rz_share:.1%})")
    elif rz_share > 0.20: justifications.append(f"High Red Zone Share ({rz_share:.1%})")
    depth = player_row['depth_multiplier']
    if depth >= 1.2: justifications.append("Primary RB1")
    elif depth >= 1.15: justifications.append("Primary WR1")
    elif depth >= 1.1: justifications.append("Primary TE1")
    vuln = player_row['def_vuln_rate']
    if vuln > 0.40: justifications.append(f"**Favorable Matchup** (vs. {player_row['opponent']} - {vuln:.0%} of TDs allowed to {player_row['position']}s)")
    elif vuln > 0.30: justifications.append(f"Good Matchup (vs. {player_row['opponent']})")
    if not justifications: return "Consistent overall usage."
    return ", ".join(justifications)

@st.cache_resource
def get_model():
    # ... (paste the entire get_model function here, unchanged)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

def display_dashboard():
    # ... (paste the entire display_dashboard function here, unchanged)
    st.title("🏈 First Touchdown Scorer Predictions")
    all_data = load_all_data()
    if not all_data:
        st.error("Could not load data. Please try re-training the model from the Model Management page.")
        return
    model = get_model()
    
    schedule = all_data['schedule']
    current_season_schedule = schedule[schedule['season'] == CURRENT_SEASON]
    past_games = current_season_schedule[pd.to_datetime(current_season_schedule['gameday']).dt.date < TODAY]
    default_week = 1 if past_games.empty else past_games['week'].max() + 1
    available_weeks = sorted(current_season_schedule['week'].unique())
    if not available_weeks:
        st.warning("No schedule data available for the current season.")
        return
    if default_week not in available_weeks: default_week = available_weeks[0]

    selected_week = st.selectbox("Select a week:", options=available_weeks, index=available_weeks.index(default_week))

    with st.spinner(f"Generating predictions for Week {selected_week}..."):
        pipeline = PredictionPipeline(all_data, selected_week, CURRENT_SEASON)
        prediction_features_df = pipeline.run()

        if prediction_features_df is None or prediction_features_df.empty:
            st.warning(f"No data available to make predictions for Week {selected_week}.")
            return

        feature_cols = ['ewma_opp_share', 'ewma_rz_share', 'depth_multiplier', 'def_vuln_rate']
        X_predict = prediction_features_df[feature_cols]
        probabilities = model.predict_proba(X_predict)[:, 1]
        prediction_features_df['raw_probability'] = probabilities
        games = prediction_features_df[['game_id', 'home_team', 'away_team']].drop_duplicates()

        for _, game in games.iterrows():
            game_df = prediction_features_df[prediction_features_df['game_id'] == game['game_id']]
            st.markdown("---")
            st.header(f"{game['away_team']} @ {game['home_team']}")
            col1, col2 = st.columns(2)
            
            for i, team_name in enumerate([game['away_team'], game['home_team']]):
                team_df = game_df[game_df['team'] == team_name].copy()
                if team_df.empty: continue
                team_total_prob = team_df['raw_probability'].sum()
                team_df['probability'] = (team_df['raw_probability'] / team_total_prob) * 100 if team_total_prob > 0 else 0
                top_players = team_df.sort_values('probability', ascending=False).head(5)
                container = col1 if i == 0 else col2
                with container:
                    st.subheader(team_name)
                    fig = px.bar(top_players, x='probability', y='full_name', orientation='h', labels={'probability': 'Probability (%)', 'full_name': ''}, text='probability')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=250, margin=dict(l=10, r=10, t=10, b=10))
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_color='#1f77b4')
                    st.plotly_chart(fig, use_container_width=True)
                    for _, player in top_players.iterrows():
                        with st.expander(f"{player['full_name']} ({player['position']}) - {player['probability']:.1f}%"):
                            st.markdown(f"**Key Factors:** {generate_justification(player)}")
                            player_history = pipeline.player_usage[pipeline.player_usage['player_id'] == player['gsis_id']].sort_values(by=['season', 'week']).tail(8)
                            if not player_history.empty:
                                player_history['week_label'] = 'Wk ' + player_history['week'].astype(str)
                                fig_trends = go.Figure()
                                fig_trends.add_trace(go.Scatter(x=player_history['week_label'], y=player_history['ewma_rz_share'], mode='lines+markers', name='Red Zone Share (EWMA)'))
                                fig_trends.add_trace(go.Scatter(x=player_history['week_label'], y=player_history['ewma_opp_share'], mode='lines+markers', name='Opportunity Share (EWMA)'))
                                fig_trends.update_layout(title="Recent Usage Trend", xaxis_title="Game Week", yaxis_title="Share %", height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                                st.plotly_chart(fig_trends, use_container_width=True)

def display_model_management():
    # ... (paste the entire display_model_management function here, unchanged)
    st.title("🎛️ Model Management")
    st.subheader("Current Model Information")
    if os.path.exists(MODEL_PATH):
        model_date = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        st.success(f"**Model active:** `td_predictor_model.json`")
        st.info(f"**Last Trained:** {model_date.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No trained model found.")
    st.markdown("---")
    st.subheader("Re-Train Model")
    st.warning("This process will re-download all nflverse data and train the model from scratch. It can take 10-15 minutes.")
    if st.button("Start Re-Training Process"):
        run_training_process()

# --- Main Application Logic ---
def main():
    # ... (paste the entire main function here, unchanged)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction Dashboard", "Model Management"])
    if not os.path.exists(MODEL_PATH) and page != "Model Management":
        st.title("👋 Welcome to the NFL TD Predictor")
        st.warning("No trained model was found. Please go to the 'Model Management' page to train the initial model.")
    else:
        if page == "Prediction Dashboard":
            display_dashboard()
        elif page == "Model Management":
            display_model_management()

if __name__ == "__main__":
    main()