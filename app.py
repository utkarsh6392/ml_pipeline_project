import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="HealthAI Pro", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

# Advanced CSS for a glass-morphism feel and clean UI
st.markdown("""
    <style>
    .main-title { font-size: 3.5rem; background: -webkit-linear-gradient(#4A90E2, #50E3C2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-weight: 800; margin-bottom: 0px;}
    .sub-title { font-size: 1.2rem; color: #a3a8b8; text-align: center; margin-bottom: 40px;}
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #4A90E2; }
    .stButton>button { background-image: linear-gradient(to right, #4A90E2, #50E3C2); color: white; border-radius: 8px; border: none; transition: 0.3s; width: 100%; font-weight: bold;}
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0px 5px 15px rgba(74, 144, 226, 0.4); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">HealthAI Predictive Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Interactive Diagnostic Analytics for Diabetes, Heart Disease & Parkinsons</p>', unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR & DATA LOADING
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2810/2810015.png", width=80)
    st.title("Control Panel")
    dataset_name = st.selectbox("1️⃣ Select Pathology", ("Diabetes", "Heart Disease", "Parkinsons"))
    st.divider()
    pipeline_step = st.radio("2️⃣ Active Module", [
        "📊 Data Overview & EDA", 
        "🧹 Preprocessing Studio", 
        "🤖 Model Training Hub"
    ])
    st.divider()
    st.caption("Powered by Streamlit & Plotly")

@st.cache_data
def load_data(dataset):
    if dataset == "Diabetes":
        df = pd.read_csv("diabetes.csv")
        target = "Outcome"
    elif dataset == "Heart Disease":
        df = pd.read_csv("heart.csv")
        target = "condition"
    else:
        df = pd.read_csv("parkinsons.csv")
        target = "status"
        if 'name' in df.columns:
            df = df.drop(columns=['name'])
    return df, target

df, target_col = load_data(dataset_name)

# ==========================================
# 3. MODULE 1: DATA OVERVIEW & EDA (Interactive)
# ==========================================
if pipeline_step == "📊 Data Overview & EDA":
    st.header(f"Exploratory Data Analysis: {dataset_name}")
    
    # Interactive Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1] - 1)
    col3.metric("Target Variable", target_col)
    col4.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()
    
    # Interactive Tabs for Data Viewing
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "🎯 Target Distribution", "🔗 Feature Correlation"])
    
    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
        
    with tab2:
        fig = px.pie(df, names=target_col, title=f"Distribution of {target_col}", hole=0.4, color_discrete_sequence=['#4A90E2', '#50E3C2'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        corr = df.corr()
        fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r', title="Interactive Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. MODULE 2: PREPROCESSING STUDIO
# ==========================================
elif pipeline_step == "🧹 Preprocessing Studio":
    st.header("Data Cleaning & Outlier Management")
    
    with st.expander("ℹ️ How does Outlier Removal work?"):
        st.write("We use the **Interquartile Range (IQR)** method. It calculates the range between the 25th and 75th percentiles and removes data points that fall exceptionally far outside this range, helping models train more accurately.")

    feature_to_plot = st.selectbox("Select Feature to Inspect", df.drop(columns=[target_col]).columns)
    
    # Plotly Interactive Boxplot
    fig = px.box(df, y=feature_to_plot, title=f"Distribution of {feature_to_plot} (Hover to see details)", points="all", color_discrete_sequence=['#FF6B6B'])
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🚀 Execute Outlier Removal Engine"):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        condition = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        df_cleaned = df[condition]
        
        st.session_state['cleaned_df'] = df_cleaned
        
        st.success("✅ Outliers successfully removed and data cached in session state!")
        col1, col2 = st.columns(2)
        col1.metric("Rows Before", df.shape[0])
        col2.metric("Rows After", df_cleaned.shape[0], delta=f"-{df.shape[0] - df_cleaned.shape[0]} removed", delta_color="inverse")

# ==========================================
# 5. MODULE 3: MODEL TRAINING HUB
# ==========================================
elif pipeline_step == "🤖 Model Training Hub":
    st.header("Machine Learning Simulator")
    
    use_df = st.session_state.get('cleaned_df', df)
    if 'cleaned_df' in st.session_state:
        st.info("Using **Cleaned Dataset** from Preprocessing Studio.")
    else:
        st.warning("Using **Raw Dataset**. Go to Preprocessing Studio to remove outliers first.")

    X = use_df.drop(columns=[target_col])
    y = use_df[target_col]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Configuration")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100.0
        random_state = st.number_input("Random State Seed", value=42)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with col2:
        st.subheader("Training Engine")
        if st.button("🧠 Train Multiple Models Automatically"):
            with st.spinner('Training models and evaluating metrics...'):
                models = {
                    "Random Forest": RandomForestClassifier(random_state=random_state),
                    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                    "Support Vector Machine": SVC(random_state=random_state),
                    "Logistic Regression": LogisticRegression(random_state=random_state)
                }
                
                results = []
                best_model_name = ""
                best_acc = 0
                best_y_pred = None
                best_model = None
                
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Accuracy": acc})
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_model_name = name
                        best_y_pred = y_pred
                        best_model = model

                st.balloons()
                st.success(f"Training Complete! Top Model: **{best_model_name}**")
                
                # Interactive Results Organization
                res_tab1, res_tab2, res_tab3 = st.tabs(["🏆 Leaderboard", "🧩 Confusion Matrix", "🌟 Feature Importance"])
                
                with res_tab1:
                    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=True)
                    fig = px.bar(results_df, x="Accuracy", y="Model", orientation='h', color="Accuracy", color_continuous_scale="Viridis", title="Model Accuracy Comparison")
                    fig.update_layout(xaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                with res_tab2:
                    cm = confusion_matrix(y_test, best_y_pred)
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"), x=['Negative', 'Positive'], y=['Negative', 'Positive'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                with res_tab3:
                    if best_model_name == "Random Forest":
                        importances = best_model.feature_importances_
                        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=True)
                        fig = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title=f"What drove the {best_model_name} decisions?")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Feature importance is not natively available for {best_model_name}. Try Random Forest for feature insights.")