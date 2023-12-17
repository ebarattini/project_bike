
### README - Predicting Bicycle Usage in Paris

#### Project Overview
This project aims to predict bicycle usage in Paris, focusing on optimizing regression models to forecast the number of bikes in use. The primary objective is to minimize the Root Mean Square Error (RMSE). Our Exploratory Data Analysis can be found [here](main/EDA). Our final model comprises of a CATboost and XGBoost stacked model, the preprocessing steps can be found [here](...), while the model itself with the final results can be found [here](...).

#### Datasets 
In addition to the core dataset for the problem, we added a number of additional datasets to enhance our model. All the data used is listed below:
- Bike counters data
    - [Train Set](main/train.parquet)
    - [Test Set](main/test.parquet)
- [French school holidays](https://www.data.gouv.fr/en/datasets/jours-feries-en-france/.): This dataset comprises of public school holidays by region. This data is chosen over public holiday dates as it more effectively captures the movement patterns of the French population, giving a better insight into regional travel and activity during these periods.
- [Lockdown data](https://www.researchgate.net/figure/Timeline-of-lockdowns-in-France-and-data-collection_fig1_356080898.): The ”Lockdown Data” dataset is a custom-created, one-hot encoded representation capturing the timeline of lockdown measures in Paris, including both full and partial lockdowns. It meticulously details the periods of school and business closures, providing a binary (0 or 1) indication for each day to signify the presence or absence of these specific lockdown measures.
- [Weather data](https://www.visualcrossing.com/weather/weather-data-services/paris/metric/2020-01-01/2023-11-29.): The weather data shows hourly observations of over 20 weather factors in Paris. It was chosen over the provided weather dataset (external data) as it provides more frequent observations and thus could improve our model’s accuracy in predicting bicycle counts at a specific hour.
- [Strikes data](https://www.cestlagreve.fr/calendrier): This dataset is custom-created, based on a calendar of planned strike dates relating to public transports in Paris. The rationale was that days with announced strikes may see higher bicycle usage as a way to circumvent them, however this correla- tion was not apparent once we explored the data, thus we decided to discard the variables in our final model.
- [Vélib Subscribers data](https://www.velib-metropole.fr/en/service): Vélib offers several pricing options: bicycles can either be used without a subscription, or a monthly plan can be purchased for regular users (over 4 trips a month). The number of subscribers to this plan was included as it could be relevant to determine the overall number of users during a period.
- [SNCF delays data](https://www.sncf.com/en/commitments/transparency/open-data): The dataset comprises monthly records of public transport delays in Paris, as reported by SNCF, offering a detailed overview of the frequency and extent of delays experienced across various modes of public transportation within the city.


