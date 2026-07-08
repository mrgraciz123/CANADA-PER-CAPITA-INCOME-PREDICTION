import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="Canada Per Capita Income Predictor",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium visual styling
st.markdown("""
    <style>
        /* Import premium font */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        
        /* Apply fonts */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: 'Space Grotesk', sans-serif;
        }
        
        /* Main Title styling */
        .title-container {
            padding: 1.5rem 0rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding: 0;
        }
        
        .sub-title {
            font-size: 1.1rem;
            color: #A0AEC0;
            font-weight: 300;
            margin-top: 0.5rem;
        }

        /* Glassmorphic Metric Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            border-color: rgba(255, 75, 75, 0.4);
            box-shadow: 0 12px 40px 0 rgba(255, 75, 75, 0.1);
        }
        
        .card-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1rem;
            color: #A0AEC0;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .card-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        
        .card-val-highlight {
            color: #FF4B4B;
        }

        /* Hero Prediction Section */
        .hero-prediction {
            background: linear-gradient(135deg, rgba(255, 75, 75, 0.08) 0%, rgba(255, 143, 143, 0.03) 100%);
            border-radius: 20px;
            border: 1px solid rgba(255, 75, 75, 0.25);
            padding: 2.2rem;
            text-align: center;
            box-shadow: 0 10px 40px 0 rgba(255, 75, 75, 0.05);
            margin-bottom: 2rem;
        }
        
        .hero-title {
            font-size: 1.1rem;
            color: #E2E8F0;
            text-transform: uppercase;
            letter-spacing: 0.15rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        
        .hero-value {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .hero-sub {
            font-size: 0.95rem;
            color: #A0AEC0;
            font-style: italic;
        }
        
        /* Math display styling */
        .math-box {
            background: rgba(0, 0, 0, 0.2);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #FF4B4B;
            margin: 1rem 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Data Loading & Preparation Helper Functions
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("canada_per_capita_income.csv")
    df.columns = [col.strip() for col in df.columns]
    return df

@st.cache_resource
def train_model(df):
    X = df[['year']]
    y = df['per capita income (US$)']
    
    # Train Linear Regression Model
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Predictions on historical data to evaluate metrics
    y_pred = reg.predict(X)
    
    # Calculate performance metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return reg, r2, mae, rmse

# Load and model
try:
    df = load_data()
    reg, r2, mae, rmse = train_model(df)
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    latest_year_data = df[df['year'] == max_year].iloc[0]
    latest_income = latest_year_data['per capita income (US$)']
except Exception as e:
    st.error(f"Error loading or modeling data: {e}")
    st.stop()

# 3. Sidebar Configuration
st.sidebar.markdown("### 🛠️ Projection Controls")
st.sidebar.markdown("Adjust controls to project per capita income for a target year.")

# Target Year Selection Inputs
target_year = st.sidebar.slider(
    "Select Projection Year",
    min_value=1970,
    max_value=2050,
    value=2020,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Parameters")
st.sidebar.info(
    f"**Slope (Coefficient):** ${reg.coef_[0]:,.2f}\n\n"
    f"**Intercept:** ${reg.intercept_:,.2f}"
)
st.sidebar.markdown(
    "<small style='color: gray;'>Model is dynamically trained on the historical dataset spanning 1970 to 2016.</small>", 
    unsafe_allow_html=True
)

# 4. Main App Layout & Header
st.markdown(
    f"""
    <div class="title-container">
        <h1 class="main-title">Canada Per Capita Income</h1>
        <p class="sub-title">🍁 Linear Regression Prediction & Historical Data Analytics Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Dynamic Prediction Logic
predict_df = pd.DataFrame({'year': [target_year]})
predicted_income = reg.predict(predict_df)[0]

# Display Hero Prediction Card
st.markdown(
    f"""
    <div class="hero-prediction">
        <div class="hero-title">Per Capita Income Projection for {target_year}</div>
        <div class="hero-value">${predicted_income:,.2f}</div>
        <div class="hero-sub">Predicted value using Simple Linear Regression model</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs Navigation
tab_dash, tab_model, tab_data = st.tabs([
    "📈 Dashboard & Visualizations", 
    "🔬 Model Performance & Analytics", 
    "🗃️ Historical Data Explorer"
])

# ================= TAB 1: DASHBOARD & FORECAST =================
with tab_dash:
    # Top Metrics Grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">Model Accuracy (R²)</div>
                <div class="card-val card-val-highlight">{r2 * 100:.2f}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">Avg. Annual Growth</div>
                <div class="card-val">${reg.coef_[0]:,.2f}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">Last Known Year ({max_year})</div>
                <div class="card-val">${latest_income:,.2f}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col4:
        diff_value = predicted_income - latest_income
        diff_percent = (diff_value / latest_income) * 100
        sign = "+" if diff_value >= 0 else ""
        color = "#2ECC71" if diff_value >= 0 else "#E74C3C"
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">Change vs {max_year}</div>
                <div class="card-val" style="color: {color};">{sign}{diff_percent:.1f}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Regression Plot (Plotly)
    st.markdown("### 📊 Linear Regression Trend & Forecast")
    
    # Generate scatter plot data and trend line data
    years_extended = np.arange(min_year, 2051)
    trend_predictions = reg.predict(pd.DataFrame({'year': years_extended}))
    
    fig = go.Figure()
    
    # Scatter plot of actual values
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['per capita income (US$)'],
        mode='markers',
        name='Actual Income',
        marker=dict(
            color='#FF4B4B',
            size=9,
            line=dict(width=1, color='#FFFFFF'),
            opacity=0.85
        ),
        hovertemplate='<b>Year: %{x}</b><br>Per Capita Income: $%{y:,.2f}<extra></extra>'
    ))
    
    # Line plot of regression line
    fig.add_trace(go.Scatter(
        x=years_extended,
        y=trend_predictions,
        mode='lines',
        name='Regression Trendline',
        line=dict(color='#8F94FB', width=2, dash='dash'),
        hovertemplate='<b>Year: %{x}</b><br>Model Projection: $%{y:,.2f}<extra></extra>'
    ))
    
    # Single highlight dot for user predicted point
    fig.add_trace(go.Scatter(
        x=[target_year],
        y=[predicted_income],
        mode='markers+text',
        name=f'Predicted {target_year}',
        marker=dict(
            color='#F1C40F',
            size=15,
            symbol='star',
            line=dict(width=1.5, color='#FFFFFF')
        ),
        text=[f"  ${predicted_income:,.0f}"],
        textposition="top center",
        textfont=dict(color='#F1C40F', size=13, family='Space Grotesk'),
        hovertemplate='<b>Target Year: %{x}</b><br>Predicted Income: $%{y:,.2f}<extra></extra>'
    ))
    
    # Style layout to be premium dark
    fig.update_layout(
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            title="Year",
            titlefont=dict(color="#A0AEC0", size=14),
            tickfont=dict(color="#CBD5E0", size=12),
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            title="Per Capita Income (US$)",
            titlefont=dict(color="#A0AEC0", size=14),
            tickfont=dict(color="#CBD5E0", size=12),
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E0", size=12)
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: MODEL PERFORMANCE =================
with tab_model:
    st.markdown("### 🔬 Linear Regression Modeling Details")
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("#### Mathematical Background")
        st.markdown(
            "The model finds the linear relationship between the **independent variable (Year)** and the **dependent variable (Per Capita Income)** using ordinary least squares regression."
        )
        
        st.markdown(
            f"""
            <div class="math-box">
                <strong>Model Equation:</strong><br>
                <code style="font-size: 1.2rem; color: #FF8F8F;">Per Capita Income (US$) = {reg.coef_[0]:,.4f} &times; Year + ({reg.intercept_:,.4f})</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.latex(r"Income = \beta_1 \cdot Year + \beta_0")
        
        st.markdown("##### Parameter Definitions:")
        st.markdown(
            f"""
            - **Coefficient (Slope $\\beta_1$):** `{reg.coef_[0]:.4f}`. For every single year that passes, Canada's per capita income is estimated to grow by **${reg.coef_[0]:,.2f}** on average.
            - **Intercept ($\\beta_0$):** `{reg.intercept_:.4f}`. The theoretical per capita income of a Canadian in Year 0.
            """
        )
        
    with col_right:
        st.markdown("#### Evaluation Metrics")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="R² Score (Coefficient of Determination)", 
                value=f"{r2:.4f}",
                help="Proportion of the variance in the dependent variable that is predictable from the independent variable (closer to 1.0 is better)."
            )
            st.metric(
                label="Mean Absolute Error (MAE)",
                value=f"${mae:,.2f}",
                help="Average absolute difference between the actual and predicted values."
            )
        with metric_col2:
            st.metric(
                label="Root Mean Squared Error (RMSE)",
                value=f"${rmse:,.2f}",
                help="Standard deviation of the residuals (prediction errors)."
            )
            st.metric(
                label="Total Records Fitted",
                value=f"{len(df)}"
            )
        
        st.markdown("---")
        st.markdown("##### Model Interpretation:")
        st.info(
            f"An **R² score of {r2:.2f}** indicates that the Year alone explains approximately **{r2*100:.1f}%** of the variation "
            f"in Canada's historical per capita income. The **RMSE of ${rmse:,.2f}** indicates that typical model predictions "
            f"deviate from actual values by about that much."
        )

# ================= TAB 3: DATASET EXPLORER =================
with tab_data:
    st.markdown("### 🗃️ Historical Dataset")
    
    col_table, col_stats = st.columns([1, 1])
    
    with col_table:
        st.markdown("#### Raw Historical Records")
        # Format columns for display
        df_display = df.copy()
        df_display['per capita income (US$)'] = df_display['per capita income (US$)'].map('${:,.2f}'.format)
        
        # Interactive Search & Filter Table
        st.dataframe(
            df_display, 
            column_config={
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "per capita income (US$)": "Per Capita Income (US$)"
            },
            use_container_width=True,
            height=400
        )
        
        # Download Data button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_data,
            file_name="canada_per_capita_income.csv",
            mime="text/csv"
        )
        
    with col_stats:
        st.markdown("#### Descriptive Statistics")
        desc_stats = df['per capita income (US$)'].describe().to_frame()
        desc_stats.columns = ['Value (US$)']
        desc_stats.loc['mean', 'Value (US$)'] = f"${desc_stats.loc['mean', 'Value (US$)']:,.2f}"
        desc_stats.loc['std', 'Value (US$)'] = f"${desc_stats.loc['std', 'Value (US$)']:,.2f}"
        desc_stats.loc['min', 'Value (US$)'] = f"${desc_stats.loc['min', 'Value (US$)']:,.2f}"
        desc_stats.loc['25%', 'Value (US$)'] = f"${desc_stats.loc['25%', 'Value (US$)']:,.2f}"
        desc_stats.loc['50%', 'Value (US$)'] = f"${desc_stats.loc['50%', 'Value (US$)']:,.2f}"
        desc_stats.loc['75%', 'Value (US$)'] = f"${desc_stats.loc['75%', 'Value (US$)']:,.2f}"
        desc_stats.loc['max', 'Value (US$)'] = f"${desc_stats.loc['max', 'Value (US$)']:,.2f}"
        desc_stats.loc['count', 'Value (US$)'] = f"{int(float(desc_stats.loc['count', 'Value (US$)']))} years"
        
        st.table(desc_stats)
        
        st.markdown("##### Historical Context & Growth:")
        # Calculate overall growth
        earliest_income = df[df['year'] == min_year].iloc[0]['per capita income (US$)']
        overall_growth = latest_income - earliest_income
        growth_pct = (overall_growth / earliest_income) * 100
        
        st.success(
            f"Between {min_year} and {max_year}, Canada's per capita income grew from **${earliest_income:,.2f}** to **${latest_income:,.2f}**, "
            f"representing a total increase of **${overall_growth:,.2f}** (**{growth_pct:.2f}%** growth) over {max_year - min_year} years."
        )
