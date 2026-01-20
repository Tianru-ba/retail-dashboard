import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

# Set up Streamlit page
st.set_page_config(page_title="Istanbul Sales Analysis", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Container styling */
    .container {
        border-radius: 10px;
        border: 1px solid #E6E9EF;
        padding: 16px;
        margin-bottom: 20px;
        background-color: #F8FAFC;
    }
    
    /* Adjust vertical spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and clean data with caching
@st.cache_data
def load_and_clean_data():
    # Load the data
    data = pd.read_csv('istanbul_sales_data.csv.csv')
    
    # Data cleaning steps
    # 1. Format invoice_date to datetime
    data['invoice_date'] = pd.to_datetime(data['invoice_date'])
    
    # 2. Remove rows with negative quantity or price
    initial_rows = data.shape[0]
    data = data[(data['quantity'] >= 0) & (data['price'] >= 0)]
    removed_rows = initial_rows - data.shape[0]
    
    # 3. Calculate total_sales
    data['total_sales'] = data['quantity'] * data['price']
    
    # 4. Check for missing values
    missing_values = data.isnull().sum().sum()
    
    return data, removed_rows, missing_values

# Create SQLite in-memory database with caching
@st.cache_resource
def create_sqlite_connection(data):
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    data.to_sql('sales', conn, index=False, if_exists='replace')
    return conn

# Load the data
data, removed_rows, missing_values = load_and_clean_data()
conn = create_sqlite_connection(data)

# Show warnings if any
if removed_rows > 0:
    st.warning(f"Removed {removed_rows} rows with negative quantity or price.")
if missing_values > 0:
    st.warning("There are missing values in the data.")
    st.write(data.isnull().sum())


# Title and introduction
st.title("Istanbul Sales Data Analysis")
st.markdown("""
This dashboard provides an interactive analysis of the Istanbul sales data. 
Use the filters in the sidebar to explore different aspects of the sales performance.
""")

# Sidebar filters
st.sidebar.header("Filters")

# Category filter
categories = data['category'].unique()
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

# Gender filter
genders = data['gender'].unique()
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)

# Payment method filter
payment_methods = data['payment_method'].unique()
selected_payment_methods = st.sidebar.multiselect("Payment Method", payment_methods, default=payment_methods)

# Shopping mall filter
malls = data['shopping_mall'].unique()
selected_malls = st.sidebar.multiselect("Shopping Mall", malls, default=malls)

# Age range filter
min_age = int(data['age'].min())
max_age = int(data['age'].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# Date range filter
min_date = data['invoice_date'].min().date()
max_date = data['invoice_date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Price strategy simulation
st.sidebar.header("价格策略模拟")
price_change = st.sidebar.slider("价格变化百分比", -50, 50, 0)

# Business Strategy Simulator
st.sidebar.header("经营策略模拟器")
aov_change = st.sidebar.slider("平均客单价变动 (%)", -30, 50, 0)
order_growth = st.sidebar.slider("订单量预期增长 (%)", -20, 100, 0)
ad_cost = st.sidebar.slider("广告投入成本 ($)", 0, 100000, 0)

# Filter the data using SQL
def filter_data_with_sql(selected_categories, selected_genders, selected_payment_methods, selected_malls, age_range, date_range):
    # Build SQL query
    query_parts = ["SELECT * FROM sales"]
    conditions = []
    
    # Add category filter
    if selected_categories:
        placeholders = ','.join(['?'] * len(selected_categories))
        conditions.append(f"category IN ({placeholders})")
    
    # Add gender filter
    if selected_genders:
        placeholders = ','.join(['?'] * len(selected_genders))
        conditions.append(f"gender IN ({placeholders})")
    
    # Add payment method filter
    if selected_payment_methods:
        placeholders = ','.join(['?'] * len(selected_payment_methods))
        conditions.append(f"payment_method IN ({placeholders})")
    
    # Add mall filter
    if selected_malls:
        placeholders = ','.join(['?'] * len(selected_malls))
        conditions.append(f"shopping_mall IN ({placeholders})")
    
    # Add age range filter
    conditions.append(f"age >= ? AND age <= ?")
    
    # Add date range filter
    conditions.append(f"invoice_date >= ? AND invoice_date <= ?")
    
    # Combine conditions
    if conditions:
        query_parts.append("WHERE")
        query_parts.append(" AND ".join(conditions))
    
    # Build final query
    query = " ".join(query_parts)
    
    # Prepare parameters
    params = []
    if selected_categories:
        params.extend(selected_categories)
    if selected_genders:
        params.extend(selected_genders)
    if selected_payment_methods:
        params.extend(selected_payment_methods)
    if selected_malls:
        params.extend(selected_malls)
    params.extend([age_range[0], age_range[1]])
    params.extend([date_range[0], date_range[1]])
    
    # Execute query and return DataFrame
    df = pd.read_sql_query(query, conn, params=params)
    # Convert invoice_date back to datetime
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    return df

# Apply filters using SQL
filtered_data = filter_data_with_sql(
    selected_categories,
    selected_genders,
    selected_payment_methods,
    selected_malls,
    age_range,
    date_range
)

# Check if filtered data is empty
if filtered_data.empty:
    st.warning('⚠️ 当前筛选条件下无数据，请重新选择商场或品类。')
    st.stop()

# Data Management section
with st.sidebar.expander("数据管理", expanded=False):
    st.subheader("数据管理")
    
    # File uploader for new sales data
    uploaded_file = st.file_uploader("上传新的销售 CSV 文件", type="csv")
    
    if uploaded_file is not None:
        # Read uploaded file
        new_data = pd.read_csv(uploaded_file)
        
        # Data cleaning for new data
        # 1. Format invoice_date to datetime
        new_data['invoice_date'] = pd.to_datetime(new_data['invoice_date'])
        
        # 2. Remove rows with negative quantity or price
        new_data = new_data[(new_data['quantity'] >= 0) & (new_data['price'] >= 0)]
        
        # 3. Calculate total_sales
        new_data['total_sales'] = new_data['quantity'] * new_data['price']
        
        # Merge with existing data
        combined_data = pd.concat([data, new_data], ignore_index=True)
        
        # Save combined data
        combined_data.to_csv('istanbul_sales_data.csv.csv', index=False)
        
        # Update SQLite database by recreating the cached connection
        conn = create_sqlite_connection(combined_data)
        
        st.success(f"成功上传并合并 {len(new_data)} 条记录！")
        st.info("应用将在刷新后使用新数据。")
    
    # Download button for current filtered data
    if not filtered_data.empty:
        # Create CSV for download
        csv = filtered_data.to_csv(index=False)
        
        st.download_button(
            label="下载当前报表数据",
            data=csv,
            file_name="istanbul_sales_report.csv",
            mime="text/csv"
        )

# Calculate metrics
total_sales = filtered_data['total_sales'].sum()
expected_revenue = total_sales * (1 + price_change / 100)
total_transactions = filtered_data.shape[0]
average_sale = filtered_data['total_sales'].mean() if total_transactions > 0 else 0

# Extract year for YoY calculation
filtered_data['year'] = filtered_data['invoice_date'].dt.year
total_2023 = filtered_data[filtered_data['year'] == 2023]['total_sales'].sum()
total_2024 = filtered_data[filtered_data['year'] == 2024]['total_sales'].sum()
yoy_growth = ((total_2024 - total_2023) / total_2023 * 100) if total_2023 > 0 else 0

# Layer 1: Metrics
st.subheader("关键指标概览")
with st.container():
    # Calculate percentage of total data
    total_data_sales = data['total_sales'].sum()
    data_percentage = (total_sales / total_data_sales * 100) if total_data_sales > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("总销售额", f"${total_sales:,.2f}", f"{data_percentage:.1f}% of total")
    col2.metric("平均客单价 (AOV)", f"${average_sale:,.2f}")
    
    # Determine color for YoY growth based on thresholds
    if yoy_growth < 5:
        yoy_color = "red"
    elif yoy_growth < 15:
        yoy_color = "orange"
    else:
        yoy_color = "green"
    
    # Use HTML to style the YoY growth metric with custom color
    with col3:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border-radius: 8px; padding: 16px; border: 1px solid #E6E9EF;">
            <div style="font-size: 14px; color: #64748B; margin-bottom: 4px;">YoY 增长率</div>
            <div style="font-size: 24px; font-weight: 600; color: {yoy_color};">{yoy_growth:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display expected revenue from price strategy simulation
    st.markdown("### 价格策略模拟")
    st.metric("预期营收", f"${expected_revenue:,.2f}", f"{price_change:+.0f}%")

# Business Strategy Simulator Results
st.subheader("经营策略模拟结果")
with st.container():
    # Calculate current metrics
    current_revenue = total_sales
    current_orders = total_transactions
    current_aov = average_sale
    
    # Calculate simulated metrics
    simulated_aov = current_aov * (1 + aov_change / 100)
    simulated_orders = current_orders * (1 + order_growth / 100)
    simulated_revenue = simulated_aov * simulated_orders
    simulated_profit = simulated_revenue - ad_cost
    current_profit = current_revenue  # Assuming no additional costs for current state
    
    # Calculate changes
    revenue_change = ((simulated_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
    profit_change = ((simulated_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
    
    # Create comparison table
    comparison_data = {
        "指标": ["当前营收", "模拟营收", "营收变化", "当前利润", "模拟利润", "利润变化", "当前订单量", "模拟订单量", "当前客单价", "模拟客单价", "广告投入"],
        "数值": [
            f"${current_revenue:,.2f}",
            f"${simulated_revenue:,.2f}",
            f"{revenue_change:+.1f}%",
            f"${current_profit:,.2f}",
            f"${simulated_profit:,.2f}",
            f"{profit_change:+.1f}%",
            f"{current_orders:,}",
            f"{simulated_orders:.0f}",
            f"${current_aov:,.2f}",
            f"${simulated_aov:,.2f}",
            f"${ad_cost:,.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Add insights based on simulation
    if simulated_profit > current_profit:
        st.success(f"✅ 模拟策略预计将增加利润 ${simulated_profit - current_profit:,.2f} ({profit_change:+.1f}%)")
        if ad_cost > 0:
            roi = ((simulated_revenue - current_revenue - ad_cost) / ad_cost * 100) if ad_cost > 0 else 0
            st.info(f"📊 广告投入 ROI: {roi:+.1f}%")
    else:
        st.warning(f"⚠️ 模拟策略预计将减少利润 ${current_profit - simulated_profit:,.2f} ({profit_change:+.1f}%)")
        st.info("建议调整参数以提高盈利能力。")

# Intelligent Insights section
st.subheader("智能洞察")
with st.container():
    # Determine the latest month in the data
    if not filtered_data.empty:
        # Extract year and month
        filtered_data['year_month'] = filtered_data['invoice_date'].dt.strftime('%Y-%m')
        latest_month = filtered_data['year_month'].max()
        
        # Filter data for the latest month
        latest_month_data = filtered_data[filtered_data['year_month'] == latest_month]
        
        if not latest_month_data.empty:
            # Calculate sales per mall for the latest month
            mall_sales = latest_month_data.groupby('shopping_mall')['total_sales'].sum().reset_index()
            avg_mall_sales = mall_sales['total_sales'].mean()
            
            # Identify abnormal malls (deviation > 20% from average)
            abnormal_malls = []
            for _, row in mall_sales.iterrows():
                mall = row['shopping_mall']
                sales = row['total_sales']
                deviation = ((sales - avg_mall_sales) / avg_mall_sales) * 100
                
                if abs(deviation) > 20:  # Threshold for abnormality
                    abnormal_malls.append((mall, sales, deviation))
            
            # Analyze each abnormal mall
            for mall, sales, deviation in abnormal_malls:
                # Get mall's data
                mall_data = latest_month_data[latest_month_data['shopping_mall'] == mall]
                
                # Calculate category distribution for the mall
                mall_category_dist = mall_data.groupby('category')['total_sales'].sum() / sales * 100
                
                # Calculate overall category distribution
                overall_category_dist = latest_month_data.groupby('category')['total_sales'].sum() / latest_month_data['total_sales'].sum() * 100
                
                # Calculate payment method distribution for the mall
                mall_payment_dist = mall_data.groupby('payment_method')['total_sales'].sum() / sales * 100
                
                # Calculate overall payment method distribution
                overall_payment_dist = latest_month_data.groupby('payment_method')['total_sales'].sum() / latest_month_data['total_sales'].sum() * 100
                
                # Find significant category deviations
                category_deviations = []
                for category in overall_category_dist.index:
                    if category in mall_category_dist:
                        diff = mall_category_dist[category] - overall_category_dist[category]
                        if abs(diff) > 10:  # Threshold for significant difference
                            category_deviations.append((category, diff))
                
                # Find significant payment method deviations
                payment_deviations = []
                for payment in overall_payment_dist.index:
                    if payment in mall_payment_dist:
                        diff = mall_payment_dist[payment] - overall_payment_dist[payment]
                        if abs(diff) > 15:  # Threshold for significant difference
                            payment_deviations.append((payment, diff))
                
                # Generate insight
                if deviation < 0:  # Poor performance
                    insight = f"检测到 {mall} 商场表现低迷，"
                    if category_deviations:
                        # Find the category with the most negative deviation
                        worst_category, worst_diff = min(category_deviations, key=lambda x: x[1])
                        insight += f"主要原因是 {worst_category} 品类占比低于大盘 {abs(worst_diff):.1f}%，"
                        insight += "建议引入高价值相关品类。"
                    elif payment_deviations:
                        worst_payment, worst_diff = min(payment_deviations, key=lambda x: x[1])
                        insight += f"主要原因是 {worst_payment} 支付方式占比低于大盘 {abs(worst_diff):.1f}%，"
                        insight += "建议优化支付流程以提升转化率。"
                    else:
                        insight += "未发现明显的品类或支付方式异常，建议全面评估运营策略。"
                else:  # Good performance
                    insight = f"检测到 {mall} 商场表现优异，"
                    if category_deviations:
                        # Find the category with the most positive deviation
                        best_category, best_diff = max(category_deviations, key=lambda x: x[1])
                        insight += f"主要原因是 {best_category} 品类占比高于大盘 {best_diff:.1f}%，"
                        insight += "建议继续加大该品类的投入。"
                    elif payment_deviations:
                        best_payment, best_diff = max(payment_deviations, key=lambda x: x[1])
                        insight += f"主要原因是 {best_payment} 支付方式占比高于大盘 {best_diff:.1f}%，"
                        insight += "建议优化该支付方式的用户体验。"
                    else:
                        insight += "各品类和支付方式分布均衡，建议保持当前策略并探索新增长点。"
                
                # Display insight
                st.info(insight)
        else:
            st.warning("当前筛选条件下无最新月份数据，无法生成智能洞察。")
    else:
        st.warning("当前筛选条件下无数据，无法生成智能洞察。")

# Layer 2: Structure Distribution
st.subheader("销售分布概览")
with st.container():
    col1, col2, col3 = st.columns(3)
    
    # Sales by category (horizontal bar chart)
    with col1:
        sales_by_category = filtered_data.groupby('category')['total_sales'].sum().sort_values(ascending=True)
        if not sales_by_category.empty:
            fig = px.bar(sales_by_category, x=sales_by_category.values, y=sales_by_category.index,
                         orientation='h',
                         color_discrete_sequence=['#636EFA', '#00CC96', '#19D3F3', '#AB63FA'])
            fig.update_traces(marker_line_width=0, opacity=0.85)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=True),
                yaxis=dict(showgrid=False, zeroline=True),
                height=350,
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sales by gender (donut chart)
    with col2:
        sales_by_gender = filtered_data.groupby('gender')['total_sales'].sum()
        if not sales_by_gender.empty:
            fig = px.pie(sales_by_gender, names=sales_by_gender.index, values=sales_by_gender.values,
                         hole=0.5, color_discrete_sequence=['#003366', '#0071BC', '#29ABE2', '#7ACCC8'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350,
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sales by payment method (donut chart)
    with col3:
        sales_by_payment = filtered_data.groupby('payment_method')['total_sales'].sum()
        if not sales_by_payment.empty:
            fig = px.pie(sales_by_payment, names=sales_by_payment.index, values=sales_by_payment.values,
                         hole=0.5, color_discrete_sequence=['#003366', '#0071BC', '#29ABE2', '#7ACCC8'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350,
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

# Layer 3: Trend Comparison
st.subheader("趋势对比分析")
with st.container():
    col_left, col_right = st.columns(2)
    
    # Yearly comparison chart
    with col_left:
        if not filtered_data.empty:
            # Calculate monthly sales by year
            filtered_data['month'] = filtered_data['invoice_date'].dt.month
            monthly_sales = filtered_data.groupby(['year', 'month'])['total_sales'].sum().reset_index()
            pivot_data = monthly_sales.pivot(index='month', columns='year', values='total_sales').fillna(0)
            
            if not pivot_data.empty:
                fig = px.line(pivot_data, x=pivot_data.index, y=pivot_data.columns,
                             labels={'x': 'Month', 'y': 'Sales'},
                             color_discrete_map={2023: '#D3D3D3', 2024: '#0071BC'})
                fig.update_traces(line=dict(width=3))
                
                # Add peak month annotations
                for year in pivot_data.columns:
                    if year in [2023, 2024]:
                        peak_month = pivot_data[year].idxmax()
                        peak_value = pivot_data[year].max()
                        fig.add_annotation(
                            x=peak_month,
                            y=peak_value,
                            text=f'Peak: {peak_month}月',
                            showarrow=True,
                            arrowhead=1
                        )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=True, tickmode='linear', dtick=1),
                    yaxis=dict(showgrid=False, zeroline=True),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Mall efficiency ranking
    with col_right:
        if not filtered_data.empty:
            mall_efficiency = filtered_data.groupby('shopping_mall')['total_sales'].sum().sort_values(ascending=True)
            if not mall_efficiency.empty:
                fig = px.bar(mall_efficiency, x=mall_efficiency.values, y=mall_efficiency.index,
                             orientation='h',
                             color_discrete_sequence=['#636EFA', '#00CC96', '#19D3F3', '#AB63FA'])
                fig.update_traces(marker_line_width=0, opacity=0.85)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=True),
                    yaxis=dict(showgrid=False, zeroline=True),
                    height=400,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

# Layer 4: Deep Dive Analysis (Tabs)
st.subheader("深度下钻探索")
with st.container():
    if not filtered_data.empty:
        # Create age groups
        age_bins = [18, 26, 36, 46, 56, 100]
        age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        filtered_data['age_group'] = pd.cut(filtered_data['age'], bins=age_bins, labels=age_labels, include_lowest=True)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(['人群渗透热力', '客单价分布', '支付效率分析'])
        
        with tab1:
            st.subheader("人群与品类热力图")
            # Heatmap
            heatmap_data = filtered_data.groupby(['age_group', 'category'], observed=True)['total_sales'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='age_group', columns='category', values='total_sales').fillna(0)
            
            if not heatmap_pivot.empty:
                # Use custom blue gradient for better contrast
                color_scale = ['#f7fbff', '#6baed6', '#08306b']
                
                # Create heatmap with annotations
                fig = px.imshow(heatmap_pivot, 
                               labels=dict(x="Category", y="Age Group", color="Sales"),
                               color_continuous_scale=color_scale)
                
                # Add text annotations with dynamic color
                for i, age_group in enumerate(heatmap_pivot.index):
                    for j, category in enumerate(heatmap_pivot.columns):
                        value = heatmap_pivot.loc[age_group, category]
                        # Calculate text color based on value (darker values get white text)
                        if value > heatmap_pivot.values.max() * 0.6:
                            text_color = 'white'
                        else:
                            text_color = 'black'
                        # Add annotation
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=f"${value:,.0f}",
                            showarrow=False,
                            font=dict(color=text_color, size=10)
                        )
                
                fig.update_traces(xgap=0.1, ygap=0.1)  # Add gap between cells
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Business conclusion
                top_age_group = filtered_data.groupby('age_group', observed=True)['total_sales'].sum().idxmax()
                top_category = filtered_data.groupby('category', observed=True)['total_sales'].sum().idxmax()
                st.write(f"业务结论：{top_age_group} 年龄段是核心客群，偏好购买 {top_category} 品类，应重点关注该群体的需求。")
        
        with tab2:
            st.subheader("客单价分布直方图")
            # Histogram
            if not filtered_data['total_sales'].empty:
                avg_order_value = filtered_data['total_sales'].mean()
                fig = px.histogram(filtered_data, x='total_sales', 
                                   labels={'x': 'Order Value', 'y': 'Frequency'},
                                   color_discrete_sequence=['#0071BC'])
                fig.add_vline(x=avg_order_value, line_dash="dash", line_color="red",
                             annotation_text=f"Average: ${avg_order_value:.2f}")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=True),
                    yaxis=dict(showgrid=False, zeroline=True),
                    height=500,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Business conclusion
                st.write(f"业务结论：客单价主要分布在中等区间，平均值为 ${avg_order_value:.2f}，建议针对高客单价客户设计专属优惠。")
        
        with tab3:
            st.subheader("支付效率分析")
            # Violin plot
            payment_data = filtered_data[filtered_data['payment_method'].isin(['Credit Card', 'Cash'])]
            if not payment_data.empty:
                fig = px.violin(payment_data, x='payment_method', y='total_sales',
                               labels={'x': 'Payment Method', 'y': 'Order Value'},
                               color='payment_method',
                               color_discrete_map={'Credit Card': '#0071BC', 'Cash': '#D3D3D3'})
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=True),
                    yaxis=dict(showgrid=False, zeroline=True),
                    height=500,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Business conclusion
                cc_avg = payment_data[payment_data['payment_method'] == 'Credit Card']['total_sales'].mean()
                cash_avg = payment_data[payment_data['payment_method'] == 'Cash']['total_sales'].mean()
                preferred_method = 'Credit Card' if cc_avg > cash_avg else 'Cash'
                st.write(f"业务结论：{preferred_method} 用户的平均订单金额更高，建议优化该支付方式的使用体验以提升客单价。")
    else:
        st.write("No data available for deep dive analysis.")

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_data)

# Add some additional insights
st.subheader("Additional Insights")
if total_transactions > 0:
    # Most popular category
    sales_by_category = filtered_data.groupby('category')['total_sales'].sum()
    most_popular_category = sales_by_category.idxmax() if not sales_by_category.empty else "N/A"
    # Most used payment method
    most_used_payment = filtered_data['payment_method'].mode().iloc[0] if not filtered_data['payment_method'].mode().empty else "N/A"
    # Busiest mall
    sales_by_mall = filtered_data.groupby('shopping_mall')['total_sales'].sum()
    busiest_mall = sales_by_mall.idxmax() if not sales_by_mall.empty else "N/A"
    
    st.markdown(f"""
    - Most popular category: **{most_popular_category}**
    - Most used payment method: **{most_used_payment}**
    - Busiest shopping mall: **{busiest_mall}**
    """)