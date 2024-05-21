# Loading all of the required libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib # Required for loading the model
import plotly.express as px # Import for the bar chart of features
import plotly.graph_objects as go
from scipy.spatial.distance import cdist # For the eclidian distance calculation and customer similarity

# Loading the models and Dataframe using joblib library and functions
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = joblib.load(file)
    return model

model = load_model("https://github.com/data-driven-prototypes-2023/s1_Olaf_Lesniak/blob/218e8da2e87bce2f78d17a59983f5afa2aa24ba9/assignment1_model3.pkl")
model_k = load_model("https://github.com/data-driven-prototypes-2023/s1_Olaf_Lesniak/blob/218e8da2e87bce2f78d17a59983f5afa2aa24ba9/full_kmeans_pipeline.pkl")

@st.cache_resource
def load_data():
    url = "https://raw.githubusercontent.com/jnin/information-systems/main/data/AI2_23_24_credit_score.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Short data cleaning and preprocessing
df.fillna(0, inplace=True)  # Set NaN values to zero
median_values = df.median()
df_types = {'INCOME': 'numeric', 'SAVINGS': 'numeric', 'DEBT': 'numeric', "CREDIT_SCORE": 'numeric',
            "CAT_MORTGAGE": 'categorical', "R_HOUSING_DEBT": 'numeric', "R_GROCERIES_SAVINGS": 'numeric'}

st.title("Credit Worthiness Prediction for Bank Customers")
tab_intro, tab_pred, tab_table = st.tabs(["Introduction 📄", "Prediction 🔮", "Table 📊"])

# Initialize the session state (used by the table tab) --> First advanced functionality of streamlit library
if 'first_click' not in st.session_state:
    st.session_state.first_click = True

# Introduction Tab
with tab_intro:
    st.markdown("""
    ## Welcome to the Credit Worthiness Prediction App! 🏦

    This is a simple Streamlit app to predict the credit worthiness of bank customers. The data used is a subset of the AI2_23_24_credit_score dataset. The dataset contains information about the income, savings, debt, and spending habits of bank customers. The goal is to predict whether a customer will default on a loan based on their credit score and spending habits.
    """)

    st.markdown("### Dataset Columns 📋")
    st.write(df.columns.to_frame(index=False, name="Features"))

    st.markdown("""
    ### Target Variable 🎯
    The target variable is **DEFAULT**, which indicates whether a customer has defaulted on a loan. The other columns are features that will be used to predict the target variable.

    ### Importance of the Model 🚀
    This is essential for any bank to predict the credit worthiness of its customers and make informed decisions about lending money. This does not only have monetary implications but also operational and reputational implications for the bank. The model can help the bank identify high-risk customers and take appropriate actions to mitigate the risk of default.

    ### Key Features 🗝️
    We have identified that the most important features to attain high accuracy are: **CREDIT_SCORE, CAT_MORTGAGE, R_HOUSING_DEBT, R_GROCERIES_SAVINGS**
    """)

    if hasattr(model, 'named_steps') and hasattr(model.named_steps['logistic_regression'], 'coef_'): # First advanced funtion to check the existence of different attributes, which will be used multiple times for applying the ML models
        estimator = model.named_steps['logistic_regression']
        coef = pd.Series(estimator.coef_.ravel(), index=estimator.feature_names_in_).sort_values()
        color_scale = px.colors.diverging.RdYlGn[::1]
        fig = px.bar(x=coef.index, y=coef.values, labels={'x': 'Feature', 'y': 'Coefficient'}, title='Feature Importance',
                     color=coef.values, color_continuous_scale=color_scale)
        fig.update_layout(yaxis=dict(range=[-0.8, 0.8], tickvals=np.arange(-0.8, 0.9, 0.2), ticktext=[f"{x:.2f}" for x in np.arange(-0.8, 0.9, 0.2)], tickformat=".2f", showline=True))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### Feature importance not available for this model.")
        
    st.markdown("""
    ### Model Selection 🧠
    I selected this dataset as it is very relevant for banks to know their customers better. Using a logistic regression for analyzing the classification and applying k-means to analyze other similar customers for further default analysis.
    
    ### Utility of the Prototype 🛠️
    The prototype is built into three tabs:
    - **Introduction Tab**: Provides an overview of the prototype.
    - **Prediction Tab 🔮**: Allows the user to input the values of the customer and predict the default probability and the cluster of the customer.
    - **Table Tab 📊**: Allows the user to filter the data based on the selected features and download the filtered table.

    This setup helps a bank agent better judge how likely a customer is to default, not solely based on logistic regression but also using k-means clustering to find similar customers. Moreover, having the functionality to filter data based on selected features and download the filtered table is essential for upper management to make the final decision on whether to approve the loan or not.
    
    ### Main Difficulties 🚧
    I found integrating k-means clustering with the logistic regression model quite challenging. I had to develop it modularly so that it could be easily integrated into the Streamlit app. 

    Often, the feature selection prediction did not work, giving errors for string parameters (especially `CAT_MORTGAGE`). Furthermore, my editor dataframe table did not work as expected as I could not filter the data based on the selected features because the default table would not persist as required or the whole table would not appear. Fortunately, I was able to fix this issue by using the session state in the beginning.
""")


# Prediction Tab
with tab_pred:
    st.markdown("## Prediction Calculator 🔮")
    input_features = ['INCOME', 'SAVINGS', 'DEBT', "CREDIT_SCORE", "R_HOUSING_DEBT", "R_GROCERIES_SAVINGS"]
    feature_medians = {feature: median_values[feature] for feature in input_features} # Second advanced function to create a dictionary with the median values of the features that are non selected

    user_input = {}

    for feature in input_features:
        user_input[feature] = st.number_input(f"Enter {feature}", value=float(feature_medians[feature]))

    if st.button("Predict"):
        updated_values = median_values.copy()
        updated_values.update(user_input)
        input_df = pd.DataFrame([updated_values], columns=median_values.index)

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)
            result = f"Probability of defaulting: {probabilities[0][1]:.2f}"
        else:
            prediction = model.predict(input_df)
            result = "Not likely to default" if prediction[0] == 0 else "Likely to default"
        st.success(result)

        st.markdown("## KMeans Clustering 🔍")
        cluster_label = model_k.predict(input_df)
        cluster_result = f"Cluster: {cluster_label[0]}"
        st.success(cluster_result)

        # Calculate default likelihood for each cluster
        df['Cluster'] = model_k.predict(df.drop(columns=['DEFAULT']))
        cluster_default_prob = df.groupby('Cluster')['DEFAULT'].mean().reset_index()
        cluster_default_prob.columns = ['Cluster', 'Default Probability']

        # Cluster descriptions using the median values per cluster
        cluster_descriptions = df.groupby('Cluster').median().reset_index()
        
        st.markdown("### Cluster Descriptions using median values per Cluster📊")
        st.write(cluster_descriptions)
        
        # Add the new customer's default probability to the cluster_default_prob DataFrame
        customer_cluster = cluster_label[0]
        customer_default_prob = probabilities[0][1] if hasattr(model, 'predict_proba') else (1 if prediction[0] == 1 else 0)
        cluster_default_prob = cluster_default_prob.append({'Cluster': customer_cluster, 'Default Probability': customer_default_prob}, ignore_index=True)

        fig = px.scatter(df, x='INCOME', y='DEFAULT', color='Cluster', title='KMeans Clustering of Customer Segments with Default Probability')
        fig.add_trace(go.Scatter(x=input_df['INCOME'], y=[customer_default_prob], mode='markers', marker=dict(color='red', size=12), name='Customer'))
        st.plotly_chart(fig) # This is the first new interactive widget allowing to plot the graph

        # Find top 5 most similar customers based on Euclidean distance
        distances = cdist(df[input_features], input_df[input_features])
        df['Distance'] = distances[:, 0]
        top_5_similar_customers = df.nsmallest(5, 'Distance')

        st.markdown("### Top 5 Most Similar Customers 🔍")

        for idx, customer in top_5_similar_customers.iterrows():
            default_status = "Defaulted" if customer['DEFAULT'] else "Not Defaulted"
            st.markdown(f"#### Customer {idx + 1} 🚀 **{default_status}**")
            cols = st.columns(len(input_features))
            for col, feature in zip(cols, input_features):
                similarity = 100 - abs((customer[feature] - input_df[feature].values[0]) / input_df[feature].values[0] * 100)
                similarity = max(0, min(100, similarity))  # Ensure similarity is between 0 and 100
                col.markdown(f"{feature}: {customer[feature]:.2f}")
                if not np.isnan(similarity):
                    col.progress(similarity / 100)

# Table Tab
# Table Tab
with tab_table:
    df = load_data()
    all_columns = df.columns.tolist()

    with st.sidebar:
        st.markdown("""
        ## Welcome to the Sidebar! 🎉
        
        Here you can adjust your values to find similar customers. If you seek to find the whole table just select empty under the dropdown menu.
        """)

        # Multi-select for feature selection
        selected_features = st.multiselect("Select Features", all_columns)
        clear_filters = st.checkbox("Clear All Filters")

        # Logic to handle feature selection and clearing filters
        if clear_filters:
            selected_columns = ["DEFAULT"]
        else:
            selected_columns = selected_features + ["DEFAULT"]

        # Ensure "DEFAULT" column is always included
        if "DEFAULT" not in selected_columns:
            selected_columns.append("DEFAULT")

        # Create sliders for the selected features
        column_range_sliders = {}
        if not clear_filters:
            for column in selected_columns:
                if column != "DEFAULT":
                    min_value = df[column].min()
                    max_value = df[column].max()
                    column_range_sliders[column] = st.slider(
                        f"Range for {column}",
                        min_value,
                        max_value,
                        (min_value, max_value)
                    ) # This is the second widget having a slider but open on both ends enabling to give in a range to filter

        show_all_features = st.button("Show All Features")

    st.markdown("## Interactive Table 📊")
    st.markdown("Here you can sort, search and filter the data based on your preferences (Pagination). You can also adjust the range sliders to filter the data based on the selected feature and even download the filtered table.")

    # Check if it's the first click, if so, display the full table
    if st.session_state.first_click:
        df_display = df
        st.session_state.first_click = False
    elif show_all_features or clear_filters:
        df_display = df
    else:
        df_display = df[selected_columns]
        for column in column_range_sliders:
            min_value, max_value = column_range_sliders[column]
            df_display = df_display[(df_display[column] >= min_value) & (df_display[column] <= max_value)]

    st.data_editor(df_display, height=700) # This is the third new interactive widget allowing to sort and download the selected table
