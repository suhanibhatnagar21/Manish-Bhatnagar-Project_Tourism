# Tourism Package Prediction - MLOps Pipeline

## Project Overview

### Business Context
Visit with Us, a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement. This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.

### Objective
As an MLOps Engineer at Visit with Us, the responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases. The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them. The pipeline includes data cleaning, preprocessing, transformation, model building, training, evaluation, and deployment, ensuring consistent performance and scalability. By leveraging GitHub Actions for CI/CD integration, the system enables automated updates, streamlines model deployment, and improves operational efficiency.

### Data Dictionary

#### Customer Details
| Feature | Description |
|---------|-------------|
| CustomerID | Unique identifier for each customer |
| ProdTaken | Target variable - purchased a package (0: No, 1: Yes) |
| Age | Age of the customer |
| TypeofContact | Method of contact (Company Invited or Self Inquiry) |
| CityTier | City category based on development and living standards (Tier 1 > Tier 2 > Tier 3) |
| Occupation | Customer occupation (Salaried, Freelancer, etc.) |
| Gender | Gender of the customer (Male, Female) |
| NumberOfPersonVisiting | Total people accompanying the customer |
| PreferredPropertyStar | Preferred hotel rating |
| MaritalStatus | Marital status (Single, Married, Divorced) |
| NumberOfTrips | Average annual trips taken |
| Passport | Valid passport holder (0: No, 1: Yes) |
| OwnCar | Car owner (0: No, 1: Yes) |
| NumberOfChildrenVisiting | Children below age 5 accompanying |
| Designation | Customer designation in their organization |
| MonthlyIncome | Gross monthly income |

#### Customer Interaction Data
| Feature | Description |
|---------|-------------|
| PitchSatisfactionScore | Customer satisfaction with the sales pitch |
| ProductPitched | Type of product pitched to the customer |
| NumberOfFollowups | Total follow-ups by salesperson after pitch |
| DurationOfPitch | Duration of the sales pitch |

## Live Demo
Streamlit App: https://huggingface.co/spaces/Suhani2128/wellness-tourism-prediction

## Project Structure
tourism_project/
├── .github/workflows/pipeline.yml
├── data/tourism.csv
├── deployment/app.py
├── deployment/Dockerfile
├── deployment/requirements.txt
├── hosting/hosting.py
├── model_building/data_register.py
├── model_building/prep.py
├── model_building/train.py
└── requirements.txt

## Pipeline Steps
| Step | Script | Description |
|------|--------|-------------|
| 1 | data_register.py | Upload raw CSV to Hugging Face dataset space |
| 2 | prep.py | Clean, encode, split and re-upload train/test |
| 3 | train.py | Train XGBoost, tune with GridSearchCV, log with MLflow, register model |
| 4 | hosting.py | Deploy Streamlit app to Hugging Face Spaces |

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.9477 |
| Precision | 0.8951 |
| Recall | 0.8258 |
| F1-Score | 0.8591 |
| ROC-AUC | 0.9696 |

Best Hyperparameters (GridSearchCV):
- n_estimators: 300
- max_depth: 7
- learning_rate: 0.2
- subsample: 1.0
- colsample_bytree: 0.8

## Technologies Used
- Python 3.9
- XGBoost - Classification model
- scikit-learn - Preprocessing and evaluation
- MLflow - Experiment tracking
- Streamlit - Web application
- Docker - Containerisation
- GitHub Actions - CI/CD pipeline
- Hugging Face Hub - Data, model and app hosting

## GitHub Secrets Required
| Secret | Description |
|--------|-------------|
| HF_TOKEN | Hugging Face write token |

## Hugging Face Resources
| Resource | Link |
|----------|------|
| Dataset | https://huggingface.co/datasets/Suhani2128/tourism-dataset |
| Model | https://huggingface.co/Suhani2128/tourism-prediction-model |
| App Space | https://huggingface.co/spaces/Suhani2128/wellness-tourism-prediction |

## Author
Suhani Bhatnagar
GitHub: https://github.com/suhanibhatnagar21

## License
MIT License
