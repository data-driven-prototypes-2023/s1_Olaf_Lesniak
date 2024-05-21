# s1_Olaf_Lesniak
Default Rate Machine Learning App (Random Forrest + K-Means)

Credit Worthiness Prediction Portal - Assignment S1
This is a Streamlit application that allows bank agents to predict credit worthiness and analyze customer data for potential loan defaults. The application includes model comparisons and leverages k-means clustering to find similar customers for further analysis.

Features
Credit Worthiness Prediction
Predict the likelihood of a customer defaulting on a loan using logistic regression.
Analyze similar customers using k-means clustering.
Insights & Analytics
Visualize and analyze customer financial data with interactive plots.
Filter and download customized datasets based on selected features.
Installation
To run this application, you'll need Python 3.8+ installed on your machine. Clone this repository and install the required libraries using the following command:

streamlit run s1_Olaf_Lesniak.py

Credit Worthiness Prediction
Navigate to the "Prediction 🔮" page.
Input customer data manually or upload a CSV file.
View the default probability prediction and the customer’s cluster assignment.
Insights & Analytics
Navigate to the "Table 📊" page.
Filter the customer data based on selected features.
Download the filtered data as a CSV file.
Adjust range sliders to fine-tune the data displayed in the interactive table.

st.markdown("""
    ### Model Selection 🎯
    I selected this dataset as it is very relevant for banks to know their customers better. Using logistic regression for analyzing the classification and applying k-means to analyze other similar customers for further default analysis.
    
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

This template provides a comprehensive overview of your Streamlit application, including installation, usage instructions, and descriptions of the features and functionality specific to your project.


Link to the video: https://urledu-my.sharepoint.com/:v:/g/personal/olafodinn_lesniak_esade_edu/EbY6nIFIe1hCoEBWVPrZLzUB_08xBPY3eW1zsN-JfcTmFw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=ea9YvX
