import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page config
st.set_page_config(page_title="Budget Buddy", page_icon="💰", layout="wide")

# Custom CSS - Black background with red accents
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF0000;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        border: 1px solid #333;
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background-color: #2a0a0a;
        border-left: 4px solid #FF0000;
        color: #ff6b6b;
    }
    
    .alert-warning {
        background-color: #2a1a0a;
        border-left: 4px solid #FF6600;
        color: #ffa500;
    }
    
    .alert-success {
        background-color: #0a2a1a;
        border-left: 4px solid #00ff00;
        color: #34d399;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
        border: 1px solid #333 !important;
    }
    
    .stTextInput input::placeholder, .stNumberInput input::placeholder {
        color: #FF0000 !important;
        opacity: 0.7;
    }
    
    .stButton button {
        background-color: #FF0000 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #CC0000 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #FF0000 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
    }
    
    /* Better text colors */
    .stMarkdown, p, span, div {
        color: #E0E0E0;
    }
    
    /* Captions and secondary text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #B0B0B0 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1a1a1a !important;
        border: 1px solid #444 !important;
        color: #E0E0E0 !important;
    }
    
    /* Button text contrast */
    .stButton button {
        color: #FFFFFF !important;
    }
    
    /* Secondary/white buttons */
    .stButton button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background-color: #E0E0E0 !important;
    }
    
    /* Tabs text */
    .stTabs [data-baseweb="tab"] {
        color: #AAAAAA !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
    }
    
    /* Dataframe headers */
    .stDataFrame {
        color: #E0E0E0 !important;
    }
    
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if 'profile_setup' not in st.session_state:
    st.session_state.profile_setup = False

# Helper Functions
def process_bank_statement(df):
    """Process uploaded bank statement CSV"""
    try:
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract day of week (0=Monday, 6=Sunday)
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfMonth'] = df['Date'].dt.isocalendar().week
        
        # Convert amount to float (handle negative numbers)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Filter only expenses (negative amounts or withdrawals)
        df_expenses = df[df['Amount'] > 0].copy()
        
        # Remove failed transactions
        df_expenses = df_expenses[df_expenses['Status'] != 'Failed']
        
        # Categorize if not already categorized
        if 'Category' not in df_expenses.columns:
            df_expenses['Category'] = 'Others'
        
        return df_expenses
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def perform_kmeans_analysis(df, n_clusters=3):
    """Perform K-Means clustering on spending data"""
    if len(df) < n_clusters:
        return df, None
    
    # Features for clustering: Amount and DayOfWeek
    features = df[['Amount', 'DayOfWeek']].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    cluster_info = []
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        avg_amount = cluster_data['Amount'].mean()
        most_common_day = cluster_data['DayOfWeek'].mode()[0] if len(cluster_data) > 0 else 0
        most_common_category = cluster_data['Category'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        cluster_info.append({
            'Cluster': i,
            'AvgAmount': avg_amount,
            'Count': len(cluster_data),
            'CommonDay': day_names[most_common_day],
            'CommonCategory': most_common_category
        })
    
    return df, pd.DataFrame(cluster_info)

def determine_personality(df, cluster_info):
    """Determine spending personality based on patterns"""
    total_spent = df['Amount'].sum()
    
    # Weekend vs Weekday spending
    weekend_spending = df[df['DayOfWeek'].isin([5, 6])]['Amount'].sum()
    weekday_spending = total_spent - weekend_spending
    
    # Category analysis
    category_totals = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    top_category = category_totals.index[0] if len(category_totals) > 0 else 'Unknown'
    
    # Determine personality
    if weekend_spending > weekday_spending * 1.3:
        return "🎉 Weekend Warrior", "You love spending on weekends! Consider saving some for weekdays."
    elif top_category == 'Food & Dining' and category_totals.iloc[0] > total_spent * 0.35:
        return "🍔 Foodie", "Food is your passion! Try cooking at home to save more."
    elif top_category == 'Gaming' and category_totals.iloc[0] > total_spent * 0.25:
        return "🎮 Gamer", "Gaming is your thing! Set a monthly gaming budget."
    elif top_category == 'Shopping' and category_totals.iloc[0] > total_spent * 0.30:
        return "🛍️ Shopaholic", "You love shopping! Try the 24-hour rule before buying."
    elif df['Amount'].std() < df['Amount'].mean() * 0.5:
        return "💰 Consistent Spender", "You spend consistently - great for budgeting!"
    else:
        return "🎯 Balanced Spender", "You have balanced spending habits. Keep it up!"

def generate_recommendations(df, monthly_allowance, savings_goal, savings_target):
    """Generate smart recommendations"""
    recommendations = []
    alerts = []
    
    total_spent = df['Amount'].sum()
    remaining = monthly_allowance - total_spent
    
    # Budget alerts
    if remaining < 0:
        alerts.append(('danger', f"⚠️ Overspent by ₹{abs(remaining):.2f}! You've exceeded your monthly allowance."))
        recommendations.append("🚨 Stop all non-essential spending immediately and review your biggest expenses")
    elif remaining < monthly_allowance * 0.1:
        alerts.append(('warning', f"⚠️ Only ₹{remaining:.2f} left! Be very careful with spending."))
        recommendations.append("⚡ Less than 10% budget remaining - stick to essentials only for the rest of the month")
    elif remaining < monthly_allowance * 0.2:
        alerts.append(('warning', f"⚠️ Only ₹{remaining:.2f} remaining from your allowance."))
        recommendations.append("⚠️ You've used 80% of your budget - be mindful of every purchase")
    
    # Category-specific recommendations
    category_totals = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    
    # Only give recommendations for categories that are actually high
    category_recs_added = 0
    
    if 'Food & Dining' in category_totals and category_totals['Food & Dining'] > monthly_allowance * 0.3:
        food_percent = (category_totals['Food & Dining'] / monthly_allowance) * 100
        recommendations.append(f"🍳 Food spending is {food_percent:.0f}% of your budget (₹{category_totals['Food & Dining']:.0f}). Pack lunch 2-3 times a week to save ₹{category_totals['Food & Dining'] * 0.2:.0f}")
        category_recs_added += 1
    
    if 'Gaming' in category_totals and category_totals['Gaming'] > monthly_allowance * 0.2:
        gaming_percent = (category_totals['Gaming'] / monthly_allowance) * 100
        recommendations.append(f"🎮 Gaming is {gaming_percent:.0f}% of budget (₹{category_totals['Gaming']:.0f}). Set a ₹{monthly_allowance * 0.15:.0f} monthly limit and explore free games")
        category_recs_added += 1
    
    if 'Shopping' in category_totals and category_totals['Shopping'] > monthly_allowance * 0.3:
        shopping_percent = (category_totals['Shopping'] / monthly_allowance) * 100
        recommendations.append(f"🛒 Shopping at {shopping_percent:.0f}% (₹{category_totals['Shopping']:.0f}). Use the 24-hour rule: wait a day before buying")
        category_recs_added += 1
    
    # If top category is taking too much budget, suggest cutting back
    if len(category_totals) > 0 and category_recs_added == 0:
        top_category = category_totals.index[0]
        top_amount = category_totals.iloc[0]
        top_percent = (top_amount / monthly_allowance) * 100
        
        if top_percent > 35:
            recommendations.append(f"💰 {top_category} is your biggest expense at {top_percent:.0f}% (₹{top_amount:.0f}). Try reducing by 20% to save ₹{top_amount * 0.2:.0f}")
    
    # Savings goal recommendations
    if savings_goal < savings_target and savings_target > 0:
        gap = savings_target - savings_goal
        progress_percent = (savings_goal / savings_target) * 100
        
        # Only show savings recommendations if gap is significant
        if gap > monthly_allowance * 0.1:  # More than 10% of monthly allowance
            recommendations.append(f"🎯 You're {progress_percent:.0f}% to your savings goal. Need ₹{gap:.0f} more - you've got this!")
            
            # Suggest practical ways to reach goal
            if len(category_totals) > 0:
                top_category = category_totals.index[0]
                potential_saving = category_totals.iloc[0] * 0.25
                
                if potential_saving >= gap * 0.5:  # If cutting top category by 25% gets you halfway there
                    recommendations.append(f"💡 Cut {top_category} by 25% (save ₹{potential_saving:.0f}) to boost savings significantly")
    
    # Positive reinforcement
    if remaining > monthly_allowance * 0.5:
        alerts.append(('success', f"✅ Excellent! You still have ₹{remaining:.0f} left (over 50% of budget)"))
        recommendations.append("🌟 Amazing restraint! You're on track for great savings this month")
    elif remaining > monthly_allowance * 0.3 and len(alerts) == 0:
        alerts.append(('success', f"✅ Good job! You have ₹{remaining:.0f} remaining"))
    
    # Weekly spending pattern - only if actually high
    if len(df) >= 7:  # Only calculate if we have at least a week of data
        weekly_avg = df.groupby(df['Date'].dt.isocalendar().week)['Amount'].sum().mean()
        weekly_target = monthly_allowance / 4
        
        if weekly_avg > weekly_target * 1.25:  # 25% over target
            overage = ((weekly_avg - weekly_target) / weekly_target) * 100
            recommendations.append(f"📊 Weekly spending is {overage:.0f}% too high (₹{weekly_avg:.0f} vs ₹{weekly_target:.0f} target). Reduce daily expenses by ₹{(weekly_avg - weekly_target)/7:.0f}")
    
    # If no specific recommendations, give general encouragement
    if len(recommendations) == 0:
        recommendations.append("🎯 Your spending looks balanced! Keep tracking to maintain good habits")
        recommendations.append("💪 Consider setting a new savings goal to challenge yourself")
    
    return recommendations, alerts

# Main App
st.markdown('<div class="main-header">💰 Budget Buddy</div>', unsafe_allow_html=True)
st.markdown("**AI-Powered Financial Insights for Teens**")

# Sidebar - Profile Setup
with st.sidebar:
    st.header("👤 Your Profile")
    
    monthly_allowance = st.number_input(
        "Monthly Allowance (₹)",
        min_value=0,
        value=5000,
        step=100,
        help="Enter your total monthly allowance from parents"
    )
    
    savings_goal = st.number_input(
        "Current Savings (₹)",
        min_value=0,
        value=1000,
        step=100,
        help="How much have you saved so far?"
    )
    
    savings_target = st.number_input(
        "Savings Target (₹)",
        min_value=0,
        value=5000,
        step=100,
        help="What's your savings goal?"
    )
    
    st.markdown("---")
    
    # View toggle
    view_mode = st.radio(
        "Dashboard View",
        ["👤 Teen View", "👨‍👩‍👧 Parent View"],
        help="Switch between teen and parent perspectives"
    )
    
    if st.button("🔄 Reset All Data"):
        st.session_state.transactions = pd.DataFrame()
        st.rerun()

# File Upload Section
st.header("📂 Upload Your Bank Statement")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=['csv', 'xlsx'],
    help="Upload your bank statement CSV"
)

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Process data
        df_processed = process_bank_statement(df)
        
        if df_processed is not None and len(df_processed) > 0:
            st.session_state.transactions = df_processed
            st.success(f"✅ Successfully loaded {len(df_processed)} transactions!")
        else:
            st.error("No valid transactions found in the file")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Manual Entry (Optional)
with st.expander("➕ Or Add Transactions Manually"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        merchant = st.text_input("Merchant", placeholder="e.g., Zomato")
    with col2:
        category = st.selectbox("Category", [
            "Food & Dining", "Shopping", "Gaming", "Entertainment",
            "Transport", "Education", "Utilities", "Others"
        ])
    with col3:
        amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0)
    with col4:
        trans_date = st.date_input("Date", value=datetime.now())
    
    if st.button("Add Transaction"):
        if merchant and amount > 0:
            new_trans = pd.DataFrame({
                'Date': [pd.to_datetime(trans_date)],
                'Description': [merchant],
                'Merchant': [merchant],
                'Category': [category],
                'Amount': [amount],
                'DayOfWeek': [trans_date.weekday()],
                'Status': ['Success']
            })
            
            if st.session_state.transactions.empty:
                st.session_state.transactions = new_trans
            else:
                st.session_state.transactions = pd.concat([st.session_state.transactions, new_trans], ignore_index=True)
            
            st.success(f"Added: {merchant} - ₹{amount}")
            st.rerun()

# Main Analysis Section
if not st.session_state.transactions.empty:
    df = st.session_state.transactions
    
    # Perform K-Means Analysis
    df_clustered, cluster_info = perform_kmeans_analysis(df)
    
    # Calculate metrics
    total_spent = df['Amount'].sum()
    remaining = monthly_allowance - total_spent
    budget_health = max(0, min(100, (remaining / monthly_allowance) * 100))
    savings_progress = (savings_goal / savings_target * 100) if savings_target > 0 else 0
    
    # Determine personality
    personality, personality_desc = determine_personality(df, cluster_info)
    
    # Generate recommendations
    recommendations, alerts = generate_recommendations(df, monthly_allowance, savings_goal, savings_target)
    
    # Display Alerts
    for alert_type, alert_msg in alerts:
        st.markdown(f'<div class="alert-box alert-{alert_type}">{alert_msg}</div>', unsafe_allow_html=True)
    
    # Key Metrics
    st.header("📊 Your Financial Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Spent",
            f"₹{total_spent:.2f}",
            delta=f"{len(df)} transactions",
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "Remaining",
            f"₹{remaining:.2f}",
            delta=f"{budget_health:.0f}% budget left",
            delta_color="normal" if remaining > 0 else "inverse"
        )
    
    with col3:
        avg_transaction = df['Amount'].mean()
        st.metric(
            "Avg Transaction",
            f"₹{avg_transaction:.2f}",
            delta=f"{personality.split()[1]}"
        )
    
    with col4:
        st.metric(
            "Savings Progress",
            f"{savings_progress:.0f}%",
            delta=f"₹{savings_goal:.0f} / ₹{savings_target:.0f}",
            delta_color="off"
        )
    
    # Teen View
    if "Teen" in view_mode:
        st.markdown("---")
        
        # Personality & Recommendations
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎭 Your Spending Personality")
            st.markdown(f"### {personality}")
            st.info(personality_desc)
            
            # Savings Goal Progress
            st.subheader("🎯 Savings Goal")
            st.progress(min(savings_progress / 100, 1.0))
            st.write(f"**₹{savings_goal:.2f}** / ₹{savings_target:.2f}")
        
        with col2:
            st.subheader("💡 Smart Recommendations")
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
            else:
                st.success("🌟 You're doing great! Keep up the good work!")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Spending Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 By Category", "📅 Daily Trend", "🤖 ML Clusters", "📝 Transactions"])
        
        with tab1:
            # Category breakdown
            category_data = df.groupby('Category')['Amount'].sum().reset_index()
            category_data = category_data.sort_values('Amount', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    category_data,
                    values='Amount',
                    names='Category',
                    title='Spending Distribution',
                    hole=0.4,
                    color_discrete_sequence=['#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC', '#FFE6E6']
                )
                fig_pie.update_layout(
                    paper_bgcolor='#000000',
                    plot_bgcolor='#000000',
                    font=dict(color='#D0D0D0'),
                    title_font_color='#FFFFFF',
                    legend=dict(
                        font=dict(color='#D0D0D0', size=12)
                    )
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(
                    category_data,
                    x='Category',
                    y='Amount',
                    title='Category Breakdown',
                    color='Amount',
                    color_continuous_scale=['#FF9999', '#FF0000']
                )
                fig_bar.update_layout(
                    paper_bgcolor='#000000',
                    plot_bgcolor='#000000',
                    font=dict(color='#FFFFFF'),
                    title_font_color='#FFFFFF'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            # Daily spending trend
            daily_spending = df.groupby('Date')['Amount'].sum().reset_index()
            
            fig_line = px.line(
                daily_spending,
                x='Date',
                y='Amount',
                title='Daily Spending Trend',
                markers=True
            )
            fig_line.update_traces(line_color='#FF0000', line_width=3)
            fig_line.update_layout(
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font=dict(color='#FFFFFF'),
                title_font_color='#FFFFFF'
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Weekly pattern
            weekly_pattern = df.groupby('DayOfWeek')['Amount'].sum().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_pattern['Day'] = weekly_pattern['DayOfWeek'].map(lambda x: day_names[x])
            
            fig_weekly = px.bar(
                weekly_pattern,
                x='Day',
                y='Amount',
                title='Spending by Day of Week',
                color='Amount',
                color_continuous_scale=['#FF9999', '#FF0000']
            )
            fig_weekly.update_layout(
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font=dict(color='#FFFFFF'),
                title_font_color='#FFFFFF'
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with tab3:
            # K-Means Clustering Visualization
            st.markdown("### 🤖 AI-Powered Spending Pattern Detection")
            st.info("Using K-Means Machine Learning algorithm to identify your spending patterns")
            
            if cluster_info is not None:
                # Display cluster information
                st.dataframe(
                    cluster_info.style.format({
                        'AvgAmount': '₹{:.2f}',
                        'Count': '{:.0f}'
                    }),
                    use_container_width=True
                )
                
                # Scatter plot of clusters
                fig_cluster = px.scatter(
                    df_clustered,
                    x='DayOfWeek',
                    y='Amount',
                    color='Cluster',
                    size='Amount',
                    hover_data=['Merchant', 'Category', 'Date'],
                    title='Spending Patterns (K-Means Clusters)',
                    labels={'DayOfWeek': 'Day of Week', 'Cluster': 'Spending Pattern'},
                    color_continuous_scale=['#FF0000', '#FF6666', '#FFCCCC']
                )
                
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig_cluster.update_xaxes(
                    tickmode='array',
                    tickvals=list(range(7)),
                    ticktext=day_names
                )
                
                fig_cluster.update_layout(
                    paper_bgcolor='#000000',
                    plot_bgcolor='#000000',
                    font=dict(color='#FFFFFF'),
                    title_font_color='#FFFFFF'
                )
                
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Explain clusters
                st.markdown("#### What these patterns mean:")
                for _, cluster in cluster_info.iterrows():
                    st.markdown(f"""
                    **Pattern {int(cluster['Cluster']) + 1}:** 
                    - Average spend: ₹{cluster['AvgAmount']:.2f}
                    - Most active on: {cluster['CommonDay']}
                    - Common category: {cluster['CommonCategory']}
                    - Transactions: {int(cluster['Count'])}
                    """)
        
        with tab4:
            # Transaction history
            st.dataframe(
                df[['Date', 'Merchant', 'Category', 'Amount']].sort_values('Date', ascending=False),
                use_container_width=True,
                hide_index=True
            )
    
    # Parent View
    else:
        st.markdown("---")
        st.subheader("👨‍👩‍👧 Parental Oversight Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Budget Utilization", f"{100 - budget_health:.0f}%")
        with col2:
            high_value_trans = len(df[df['Amount'] > avg_transaction * 2])
            st.metric("High-Value Transactions", high_value_trans)
        with col3:
            daily_avg = df.groupby('Date')['Amount'].sum().mean()
            st.metric("Daily Average", f"₹{daily_avg:.2f}")
        
        # Detailed breakdown
        st.markdown("### 📊 Detailed Category Analysis")
        category_summary = df.groupby('Category').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        category_summary.columns = ['Total Spent', 'Avg Transaction', 'Count']
        category_summary = category_summary.sort_values('Total Spent', ascending=False)
        st.dataframe(category_summary, use_container_width=True)
        
        # All transactions
        st.markdown("### 📝 Complete Transaction History")
        st.dataframe(
            df[['Date', 'Merchant', 'Category', 'Amount']].sort_values('Date', ascending=False),
            use_container_width=True,
            hide_index=True
        )

else:
    # Welcome screen
    st.info("👆 Upload your bank statement or add transactions manually to get started!")
    
    st.markdown("""
    ### 🚀 How to use Budget Buddy:
    
    1. **Upload Bank Statement**: Click the upload button and select your CSV/Excel file
    2. **Or Add Manually**: Expand the manual entry section to add transactions one by one
    3. **Set Your Profile**: Enter your monthly allowance and savings goal in the sidebar
    4. **Get Insights**: View AI-powered recommendations and spending analysis
    5. **Track Progress**: Monitor your savings goal and budget health
    
    ### 🤖 What you'll get:
    - **Smart Recommendations** based on your spending patterns
    - **ML-Powered Personality Detection** using K-Means clustering
    - **Budget Alerts** when you're overspending
    - **Visual Analytics** with interactive charts
    - **Parent Dashboard** for oversight
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 Powered by K-Means Machine Learning | Built with Streamlit</p>
    <p>Smart Financial Analytics for Teens</p>
</div>
""", unsafe_allow_html=True)
