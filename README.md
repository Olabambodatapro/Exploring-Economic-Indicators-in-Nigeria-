# Exploring-Economic-Indicators-in-Nigeria-

Project Title: GDP, Growth, and Inflation Analysis with Predictive Modeling
________________________________________
1. Project Overview
•	Objective: Analyze GDP, GDP growth rate, and inflation over the years, and predict future GDP values using linear regression.
•	Tools and Libraries Used: Python, matplotlib for data visualization, scikit-learn for model building, and numpy for data manipulation.
________________________________________
2. Data Collection & Preprocessing
•	Data Source: World Development Indicators | DataBank (worldbank.org)
•	Variables Analyzed:
o	GDP: Gross Domestic Product over the years.
o	GDP Growth: Annual growth percentage in GDP.
o	Inflation: Inflation rate as GDP deflator.
•	Data Preparation Steps: Briefly mention any preprocessing steps like handling missing values, converting data types, and structuring data for time series analysis.
________________________________________
3. Exploratory Data Analysis (EDA)
•	GDP Over the Years: Presented in blue, shows the overall economic trend. Describe any notable patterns (e.g., a sharp rise after 2005).
•	GDP Growth (Annual %) Over the Years: Presented in green, highlights changes in economic growth rates. Discuss any periods of growth or decline.
•	Inflation (Annual %) Over the Years: Presented in red, depicts inflation trends. Mention any significant peaks or troughs and their possible implications.
•	Visualization Screenshots: (Refer to the attached image with side-by-side graphs for GDP, GDP growth, and inflation).
 
________________________________________
4. Predictive Modeling
•	Model Chosen: Linear Regression, chosen for simplicity in forecasting economic trends.
•	Error Metric: Root Mean Squared Error (RMSE) = 199,563,875,987.14. This metric was used to evaluate the model's accuracy.
•	Future Prediction:
o	Target Year: 2025
o	Predicted GDP for 2025: -7.39 × 10¹⁸ USD
•	Interpretation: The negative predicted GDP for 2025 indicates potential issues with the model, possibly due to outliers, limited data, or linear regression’s inability to capture complex economic patterns.
________________________________________
5. Observations and Analysis
•	GDP Trends: Analysis suggests rapid growth in recent years, with some fluctuations.
•	GDP Growth Patterns: GDP growth shows volatility; discuss any economic or global events that might align with peaks or declines.
•	Inflation Insights: Inflation rates show notable peaks, possibly related to economic booms or financial crises.
________________________________________

6. Model Evaluation and Limitations
•	Model Evaluation:
o	RMSE provides an error measure but shows a high error value, indicating that the model’s predictions may lack accuracy.
•	Limitations:
o	Linear Regression: The model may not adequately capture non-linear trends in economic data.
o	Negative Prediction: The 2025 prediction may be due to model limitations, requiring more sophisticated models or more extensive data.
________________________________________
7. Future Improvements
•	Consider using more advanced models, such as Polynomial Regression, Time Series models (ARIMA), or machine learning algorithms that handle non-linear patterns better.
•	Gather more historical data to improve model robustness and accuracy.
•	Explore the inclusion of additional economic indicators like unemployment rates or investment data.
________________________________________
8. Conclusion
•	Summary: The analysis successfully visualized GDP, GDP growth, and inflation trends over time, highlighting significant economic patterns.
•	Insights: The data indicates an overall growth in GDP with fluctuations in growth rate and inflation, but limitations in predictive modeling suggest caution in using simple linear regression for complex economic forecasting.
•	Next Steps: Future work may involve more sophisticated models to improve prediction accuracy.
