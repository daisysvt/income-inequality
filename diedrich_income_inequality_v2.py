# Imports
import streamlit as st
import statsmodels.api as sm
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cspp_data_2023-12-06.csv')

df = load_data()

# Select features and target variable
features = ['exp_public_welfare', 'welfare_spending_percap', 'z_tanf_initialelig', 'z_ssi_afdc_families_payments', 'x_chip_pregnantwomen']
target = 'gini_coef'

# Create a subhead for the table of contents
st.sidebar.title('Table of Contents')

# Create links to jump to the corresponding sections
st.sidebar.markdown(f"""
- [About the Issue](#about-the-issue)
- [How Can Income Inequality Be Measured?](#how-can-income-inequality-be-measured)
- [Objective of the Web App](#objective-of-the-web-app)
- [Model Results](#model-results)
- [Conclusions](#conclusions)
""", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.title('User Input')

# State selection dropdown
selected_state = st.sidebar.selectbox('Select State', df['state'].unique())

# Filter the data based on the selected state
state_df = df[df['state'] == selected_state]

# Drop rows with NaN values in the target variable
state_df = state_df.dropna(subset=[target])

X = state_df[features]
y = state_df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a HistGradientBoostingRegressor
@st.cache_data
def train_model(X_train, y_train):
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Make predictions
@st.cache_data
def make_predictions(_model, X_test):
    return _model.predict(X_test)

predictions = make_predictions(model, X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Scatter plot for actual vs. predicted values with a trendline
fig1 = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, trendline='ols')
 
# Title and caption
st.title('How do State Welfare Policies Influence Income Inequality?')
st.caption('Created By: Graham Diedrich [(About Me!)](https://www.grahamdiedrich.com)', unsafe_allow_html=True)

# Feature sliders
exp_welfare = st.sidebar.slider('Public Welfare Expenditure ($)', min_value=df['exp_public_welfare'].min(), max_value=df['exp_public_welfare'].max())
welfare_spending = st.sidebar.slider('Welfare Spending per Capita ($)', min_value=df['welfare_spending_percap'].min(), max_value=df['welfare_spending_percap'].max())
z_tanf_initialelig = st.sidebar.slider('TANF Initial Eligibility Level for Family of Three ($)', min_value=df['z_tanf_initialelig'].min(), max_value=df['z_tanf_initialelig'].max())
z_ssi_afdc_families_payments = st.sidebar.slider('Average Level of AFDC Benefits per Family ($)', min_value=df['z_ssi_afdc_families_payments'].min(), max_value=df['z_ssi_afdc_families_payments'].max())
x_chip_pregnantwomen = st.sidebar.slider('CHIP Eligibility Level for Pregnant Women ($)', min_value=df['x_chip_pregnantwomen'].min(), max_value=df['x_chip_pregnantwomen'].max())

# Add a picture below the title
image_url = 'https://si-interactive.s3.amazonaws.com/prod/plansponsor-com/wp-content/uploads/2023/09/13164528/PS-091323-Income-Inequality-and-Social-Security-Solvency-1296162374-web.jpg'
st.image(image_url, use_column_width=True)

# Subtitle
st.header("About the Issue")

# Paragraph about income inequality
st.write("""
📊 Income inequality in the United States is a complex and pervasive issue that has garnered increasing attention in recent decades. The wealth gap between the rich and the poor has widened, raising concerns about **economic disparities and social justice**. This phenomenon is not only evident in statistical measures but also has [real-world implications](https://www.nytimes.com/2019/09/10/us/politics/gao-income-gap-rich-poor.html), affecting access to education 🏫, healthcare 🏥, and economic opportunities 💸.

💰 The consequences of income inequality extend beyond individual financial situations, influencing societal stability and overall well-being. Addressing these disparities requires a comprehensive understanding of the contributing factors, including [policy decisions](https://www.imf.org/en/Blogs/Articles/2017/10/11/inequality-fiscal-policy-can-make-the-difference), systemic issues, and socio-economic dynamics. As we explore the relationship between state welfare policies and income inequality, it is **crucial to consider the broader context** of this multifaceted challenge.
""")

# Subtitle
st.header("How Can Income Inequality Be Measured?")

# Paragraph about Gini coefficient
st.write("""
📐 One way to measure income inequality is the [Gini coefficient](https://ourworldindata.org/what-is-the-gini-coefficient). It represents the distribution of income among the residents of a country, providing insights into the economic disparity.
A Gini coefficient of **0 indicates perfect equality** (everyone has the same income), while a coefficient of **1 represents maximum inequality** (one person or group has all the income).""")

image_url = 'https://miro.medium.com/v2/resize:fit:1190/0*RCLJP5gEy1YCA-tu'
st.image(image_url, use_column_width=True)

st.write("""
📈 The visualization below examines the U.S. Gini coefficient over time since 1963. As you can see, the **inequality has generally increased since the 1980s**.""")

# Load dataset
dfv = pd.read_csv('SIPOVGINIUSA.csv')

# Create an interactive line graph
fig2 = px.line(dfv, x='year', y='gini', 
              labels={'gini': 'Gini Coefficient'}, 
              title='U.S. Gini Coefficient, 1963-2021',
              template='plotly', 
              line_shape='linear', 
              color_discrete_sequence=['green'])

# Update layout to adjust top and bottom margins
fig2.update_layout(margin=dict(t=35, b=20))

# Show the plot directly
st.plotly_chart(fig2)

# Paragraph about differences in Gini coefficient between U.S. states
st.write("💰 Income inequality across U.S. states exhibits notable variations, and understanding these differences involves a complex interplay of economic, social, and policy factors. **One prominent factor influencing income inequality is the diverse landscape of public spending and welfare policies implemented at the state level.**")

st.write("🏦 States with [robust public welfare programs](https://www.route-fifty.com/management/2021/09/states-weakest-social-safety-nets/185519/) and targeted spending on social services often experience lower levels of income inequality. These policies aim to provide a safety net for vulnerable populations, reducing the wealth gap and fostering economic inclusivity. 📚 🏥 For instance, states that invest significantly in public education, healthcare, and social assistance programs tend to see a more equitable distribution of income.")

st.write("📉 Conversely, states with limited welfare policies may face higher levels of income inequality. Gaps in access to [quality education, healthcare, and social support](https://nces.ed.gov/programs/maped/storymaps/edinequality/) can contribute to disparities in economic outcomes. Additionally, variations in taxation policies and social assistance program effectiveness play a crucial role in shaping the income distribution within each state.")

# Find the latest year for each state with non-NaN gini_coef
latest_year_non_nan = df.dropna(subset=['gini_coef']).groupby('st')['year'].max().reset_index()

# Merge with the original dataframe to get the corresponding Gini coefficient values
df_latest_non_nan = pd.merge(latest_year_non_nan, df, on=['st', 'year'], how='inner')

# Select the rows that are the latest year with non-NaN gini_coef
df_latest_non_nan_rows = df_latest_non_nan[df_latest_non_nan['year'] == df_latest_non_nan['year']]

# Drop all columns except 'st', 'year', and 'gini_coef'
df_latest_non_nan_selected = df_latest_non_nan_rows[['st', 'year', 'gini_coef']]

# Create an interactive choropleth map with px
fig_map = px.choropleth(
    df_latest_non_nan_selected,
    locations='st',
    locationmode='USA-states',
    color='gini_coef',
    color_continuous_scale='Viridis',
)

# Update layout for better visualization
fig_map.update_geos(scope='usa')
fig_map.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),  # Adjust the top margin to make space for the title
    title='Gini Coefficient by U.S. State, 2013',
)

# Show the map directly
st.plotly_chart(fig_map)

# Section about the Objective
st.header('Objective of the Web App')

st.write(
    "🎯 The objective of this web app is to provide users with the ability to explore and understand "
    "the relationship between program expenditures, welfare benefit levels, and the resulting impact "
    "on the Gini coefficient across U.S. states. 🔄 Users can interactively adjust parameters related to "
    "public spending and welfare policies to observe how these changes influence income inequality."
)

st.write(
    "🤔 Through this interactive platform, users can gain insights into the potential effects of policy decisions "
    "on the distribution of income within states. Adjusting variables such as public welfare expenditure, "
    "welfare spending per capita, and other relevant factors allows users to visualize the dynamics of income "
    "inequality and make informed assessments about the impact of policy choices on societal well-being."
)

# Make predictions based on user input
@st.cache_data
def predict_with_user_input(_model, user_input):
    return _model.predict([user_input])[0]

prediction = predict_with_user_input(model, [exp_welfare, welfare_spending, z_tanf_initialelig, z_ssi_afdc_families_payments, x_chip_pregnantwomen])

# Section about the results
st.header('Model Results')

# Display the predicted Gini coefficient in a large box
st.subheader('Predicted State Gini Coefficient')
predicted_gini = model.predict([[exp_welfare, welfare_spending, z_tanf_initialelig, z_ssi_afdc_families_payments, x_chip_pregnantwomen]])[0]
st.markdown(f'<div style="font-size: 24px; border: 2px solid black; padding: 10px; border-radius: 10px;">{predicted_gini:.4f}</div>', unsafe_allow_html=True)

# Dynamic paragraph with added padding to the top
st.markdown(f"""
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 20px; margin: 20px;">
        <p style="font-size: 16px; color: #333;">
            With a state public welfare expenditure of ${exp_welfare:,.2f},
            welfare spending per capita at ${welfare_spending:,.2f},
            TANF eligibility at {z_tanf_initialelig:.2f},
            AFDC benefits at {z_ssi_afdc_families_payments:.2f},
            and CHIP eligibility for pregnant women at {x_chip_pregnantwomen:.2f},
            the predicted Gini coefficient for {selected_state} is <strong>{predicted_gini:.4f}</strong>.
        </p>
    </div>
""", unsafe_allow_html=True)

# Update layout to include title
fig1.update_layout(
    title='Actual vs. Predicted Values',
)

# Update the color of the trendline and points to green
fig1.update_traces(
    line=dict(color='green'),  # Set the color of the trendline
    marker=dict(color='green'),  # Set the color of the points
)

# Display interactive plots
st.plotly_chart(fig1)

# Display evaluation metrics table
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
evs = explained_variance_score(y_test, predictions)
max_err = max_error(y_test, predictions)

st.header('Model Evaluation')
metrics_data = {'Mean Squared Error': [mse], 'R-squared': [r2], 'Mean Absolute Error': [mae], 'Explained Variance Score': [evs], 'Max Error': [max_err]}
metrics_df = pd.DataFrame(metrics_data)
st.table(metrics_df)

st.header('Conclusions')
st.write("🤝 💬 In conclusion, this exploration sheds light on the intricate dynamics of income inequality across U.S. states, revealing the multifaceted influences of public welfare policies. The disparities in wealth distribution underscore the importance of targeted interventions and strategic policy decisions. By examining the interplay of factors such as public welfare expenditure, social assistance programs, and their impact on the Gini coefficient, we gain valuable insights into the **potential avenues for addressing income inequality**. Moving forward, fostering collaboration between policymakers, researchers, and the public is crucial to designing and implementing effective policies that promote economic inclusivity. This web app serves as a tool for users to navigate through these complexities, offering a **nuanced understanding of the factors contributing to income inequality** and encouraging informed conversations about its solutions.")

# Provide a link to download the dataset
st.sidebar.markdown('[Download Dataset](https://cspp.ippsr.msu.edu)')