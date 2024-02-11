![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Fast_API_Framework-green?logo=fastapi&logoColor=white)![Docker](https://img.shields.io/badge/Docker-Containerization-blue?logo=docker&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Continuous_Integration-orange?logo=github-actions&logoColor=white) ![Amazon Web Services](https://img.shields.io/badge/AWS-Amazon_Web_Services-orange?logo=amazon-aws&logoColor=white) ![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-blue?logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-lightblue?logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-lightblueblue?logo=scikit-learn&logoColor=orange) ![Seaborn](https://img.shields.io/badge/Seaborn-Data_Visualization-yellow?) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Visualization-blue?)

# TRAVEL INSURANCE PREDICTION

This project aims to develop a predictive model for ExploreXperience, a tour company offering travel insurance, to identify customers more likely to purchase insurance post the Covid-19 pandemic. By analyzing past sales data, the project leverages machine learning techniques to create a model that predicts customer behavior and preferences.

The project structure includes:

* Data Collection and Preprocessing: The project retrieves data from an S3 bucket, preprocesses it, and saves it as CSV files. This involves handling missing values, converting data types, and scaling numeric features.
* Model Training and Evaluation: The main functionality involves training a Random Forest classifier using GridSearchCV for hyperparameter tuning. The model is evaluated using accuracy metrics and tuned for better performance.
* Model Deployment and Tracking: The trained model is deployed as a FastAPI endpoint using Docker. MLflow is used for model tracking and deployment. Model parameters, metrics, and the trained model itself are logged for reproducibility and monitoring.
* Visualization: The project includes a Jupyter Notebook for data exploration and visualization.
* Application Deployment: The project provides a FastAPI web application for deploying the trained model in a production environment. Deployment to DockerHub is automated using GitHub Actions workflows.

### The following are some of the demos for the github actions workflow run and mlflow ui and FastApi swagger demo:

![github actions](https://github.com/franciskyalo/travelinsuranceprediction/assets/94622826/8e8d0d89-738e-4273-9e77-37575d22216a)


![fast api](https://github.com/franciskyalo/travelinsuranceprediction/assets/94622826/1290dc42-7989-4418-9d8e-680b4aa8e154)



![prediction](https://github.com/franciskyalo/travelinsuranceprediction/assets/94622826/df1c041c-cb3e-4932-a7a8-e4f2b92822b4)



![mlfow ui](https://github.com/franciskyalo/travelinsuranceprediction/assets/94622826/4f46745a-af4a-45da-9c4a-a9422d82eb8f)



# RECOMMENDATIONS 

* Target the Private Sector: Since a significant portion of customers who purchased travel insurance were from the private sector (28%), ExploreXperience should focus their marketing efforts on this demographic. However, it's also important to note that a substantial number of private sector customers (42.63%) did not buy insurance. Therefore, tailored marketing strategies should be employed to address the concerns or preferences of this group.

* Leverage Education Level: Customers who were graduates were more likely to purchase travel insurance (30.75%) compared to those who were not graduates (4.98%). This indicates that education level may be a significant factor in a customer's decision to buy insurance. ExploreXperience can create targeted campaigns or incentives to attract and convert more non-graduate customers.

* Consider Chronic Disease Status: ExploreXperience should pay attention to customers with chronic diseases. Currently, only 10% of customers with chronic diseases purchased travel insurance. This represents an opportunity to provide tailored insurance offerings or additional benefits for this group. Additionally, efforts can be made to address concerns or reservations that may be preventing them from purchasing insurance.

* Engage Frequent Flyers: Given that only 12% of customers identified as frequent flyers purchased travel insurance, ExploreXperience can create promotions or packages specifically designed to cater to this group. Highlighting the benefits of insurance for frequent travelers and providing attractive offers can increase conversion rates.

* Promote Travel Abroad Experiences: Customers who have traveled abroad in the past are more likely to purchase travel insurance (15%). ExploreXperience can capitalize on this by emphasizing the value of insurance for international travel. This could include features like medical coverage, trip cancellation protection, and assistance services tailored for overseas trips.

* Educate Non-Travelers on Insurance Benefits: There is a large segment of customers (60%) who have not traveled abroad and have not purchased travel insurance. ExploreXperience can run awareness campaigns highlighting the importance of insurance, even for domestic trips. This can include coverage for unexpected events like trip cancellations, medical emergencies, or lost luggage.

* Customer Feedback and Customization: Gathering feedback from customers who did not purchase insurance can provide valuable insights into their concerns or reasons for not opting for it. This information can be used to customize insurance offerings, address objections, and improve the overall value proposition.

 
To clone this repo, use the code below:
```
git clone https://github.com/franciskyalo/streamlit_dashboard.git
```

Set up a virtual enviroment;

```
# On macOS and Linux
python3 -m venv venv

# On Windows
python -m venv venv

```


Activate the virtual enviroment using;

```
# On macOS and Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate

```

Install dependencies using;

```
pip install -r requirements.txt

```




