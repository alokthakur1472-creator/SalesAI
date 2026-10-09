\# SalesAI – Sales Forecasting and Business Intelligence Dashboard

**Live Dashboard:** [Open SalesAI Dashboard](https://salesai-4pclppnmbenqwiakt9uexy.streamlit.app/)

An end-to-end sales forecasting and business intelligence project that analyzes historical store sales and predicts future sales using machine learning.

SalesAI is a machine-learning-based sales forecasting and business intelligence project. It analyzes historical sales data, identifies sales trends, compares store performance, and forecasts future sales.

I developed this project to gain practical experience in Python, data analysis, machine learning, and dashboard development. The aim is to transform raw sales data into useful insights that can support better business decisions.

\## Key Features

\* Historical sales analysis and visualization.

\* Data cleaning and feature engineering.

\* Monthly sales trend analysis.

\* Store-wise performance comparison.

\* Holiday sales analysis.

\* Machine-learning-based sales forecasting.

\* Forecast evaluation using performance metrics.

\* Business intelligence dashboard with key performance indicators.



\## Technologies Used



\* Python

\* Pandas and NumPy

\* Scikit-learn

\* XGBoost

\* Streamlit

\* Plotly and Matplotlib

\* Machine Learning



\## Project Structure



```text

SalesAI/

├── dashboard/

│   └── app.py

├── models/

│   ├── best\_sales\_model.pkl

│   └── feature\_list.txt

├── outputs/

│   └── eda/

├── src/

│   ├── data\_pipeline.py

│   ├── eda.py

│   ├── forecasting.py

│   └── future\_forecast.py

└── README.md

```



\## How the Project Works



First, the sales data is cleaned and prepared for analysis. Next, exploratory data analysis is performed to identify trends and compare store performance. A machine learning model is used to forecast future sales, and the results are presented through a dashboard for easier interpretation.



\## How to Run



1\. Install Python on your computer.



2\. Clone or download this repository.



3\. Open CMD or a terminal in the project folder.



4\. Install the required libraries:



&#x20;  `pip install pandas numpy scikit-learn xgboost streamlit plotly matplotlib`



5\. Start the dashboard:



&#x20;  `streamlit run dashboard/app.py`



Ensure that the required dataset, trained model, and any additional dependencies used by the code are available in their expected locations.



\## What I Learned



Through this project, I gained practical experience in data preprocessing, exploratory data analysis, feature engineering, machine learning, forecasting, and dashboard development. I also learned how to organize different components into a complete data analytics project.



\## Future Improvements



\* Add more forecasting models for comparison.

\* Improve interactive dashboard filters.

\* Add more business performance metrics.

\* Deploy the dashboard online for easier access.



\## Author



\*\*Alok Thakur\*\*



GitHub: https://github.com/alokthakur1472-creator



This project was developed as a practical learning project in data analytics, machine learning, and business intelligence.



