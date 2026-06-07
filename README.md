# Tourism Package Prediction - MLOps Pipeline

## Project Overview
End-to-end MLOps pipeline predicting whether a customer will purchase the **Wellness Tourism Package** for Visit with Us travel company.

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
