# Exploring-Economic-Indicators-in-Nigeria-

Project Title: GDP, Growth, and Inflation Analysis with Predictive Modeling
________________________________________
1. Project Overview
- Objective: Analyze GDP, GDP growth rate, and inflation over the years, and predict future GDP values using linear regression.
- Tools and Libraries Used: Python, matplotlib for data visualization, scikit-learn for model building, and numpy for data manipulation.
________________________________________
Data Collection & Preprocessing
- Data Source: World Development Indicators | DataBank (worldbank.org)
- Variables Analyzed:
 - GDP: Gross Domestic Product over the years.
 - GDP Growth: Annual growth percentage in GDP.
 -	Inflation: Inflation rate as GDP deflator.
-	Data Preparation Steps: Briefly mention any preprocessing steps like handling missing values, converting data types, and structuring data for time series analysis.
Project Summary: Economic Data Analysis and Forecasting Using Python
This project aimed to analyze and forecast key economic indicators—specifically, Gross Domestic Product (GDP), GDP growth rate, and inflation—using Python. By examining historical trends and making future projections, I sought to provide insights into the economic landscape, identify patterns, and assess the feasibility of predictive modeling for GDP.
2. Exploratory Data Analysis (EDA) and Visualization

To gain an initial understanding of the trends, I visualized each economic indicator over time using Matplotlib.
The GDP Over the Years chart revealed a general upward trend, showing the economic growth trajectory.
The GDP Growth (Annual %) Over the Years plot illustrated periods of expansion and contraction, highlighting economic cycles and volatility.
The Inflation (Annual %) Over the Years chart helped identify fluctuations in inflation, showing peaks and declines that correlate with economic activity.
These visualizations provided critical insights into the economic landscape, allowing me to make data-driven observations on historical performance.
3. Predictive Modeling

I constructed a linear regression model to forecast future GDP values based on historical trends. After fitting the model, I assessed its performance by calculating the Root Mean Squared Error (RMSE), which resulted in a value of approximately 199.6 billion.
Using this model, I predicted the GDP for the year 2025, which yielded a value of -7.39e+18 USD. This extreme and unrealistic result pointed out limitations in the linear model, indicating either data variability that couldn’t be captured by linear regression or possible model overfitting/underfitting issues.
4. Interpretation of Results and Insights

The high RMSE and the implausible GDP forecast for 2025 highlighted the limitations of using simple linear regression on complex, non-linear economic data. Economic indicators like GDP are influenced by numerous factors, including political changes, global market dynamics, and unexpected events (e.g., pandemics), which linear models may not fully capture.
The project underscored the importance of selecting models suited for complex economic forecasting. A more sophisticated approach, such as time-series analysis with ARIMA or using machine learning models like XGBoost, may provide greater accuracy for forecasting.
________________________________________
8. Conclusion
•	Summary: The analysis successfully visualized GDP, GDP growth, and inflation trends over time, highlighting significant economic patterns.
•	Insights: The data indicates an overall growth in GDP with fluctuations in growth rate and inflation, but limitations in predictive modeling suggest caution in using simple linear regression for complex economic forecasting.
