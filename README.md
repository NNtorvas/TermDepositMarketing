## Background:

We are a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.

We are interested in developing a robust machine learning system that leverages information coming from call center data.

Ultimately, we are looking for ways to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

## Data Description:

The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

### Attributes:

- age : age of customer (numeric)

- job : type of job (categorical)

- marital : marital status (categorical)

- education (categorical)

- default: has credit in default? (binary)

- balance: average yearly balance, in euros (numeric)

- housing: has a housing loan? (binary)

- loan: has personal loan? (binary)

- contact: contact communication type (categorical)

- day: last contact day of the month (numeric)

- month: last contact month of year (categorical)

- duration: last contact duration, in seconds (numeric)

- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

### Output (desired target):

- y - has the client subscribed to a term deposit? (binary)

## Summary of the Project:

Our Analysis was conducted following these steps:

1. Import Libraries and Load Data
2. Data preprocessing
3. Exploratory Data Analysis (EDA) - get insights for the company, handle imbalanced dataset and categorical features
4. Create a base model using Lazypredict library for testing multiple model
5. Feature Importance
6. Train Final Model

### Inshights:

* The dataset was clean with no missing, null or duplicated values.
* There were some clients, who subscripted with negative balance. This does not make sense and maybe is a mistake on these observations of the dataset.
* The majority of those who have defaulted before or have a personal loan or have a housing loan have not subscribed to a term deposit.

We also saw that clients are more likely to subscribe if: 

1) have tetriary education
2) are student
3) are having a cellular contact with the company
4) their last contact month: october
5) they are single (marital)

### Conclusion:

* As we saw the duration was the most important feature in predicting if the client will subsribe or not. The client who would want to buy is likely to stay longer on the call in order to know more about the program.
And the company should persuade the client with last call duration of 11 - 35 minutes. 
* Also, our model with 90% accuracy on predicting the client's subscription or not will save them time and money.