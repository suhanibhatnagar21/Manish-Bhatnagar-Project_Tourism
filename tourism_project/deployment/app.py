import os
import pickle
import warnings
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title  = 'Tourism Package Predictor',
    page_icon   = '✈️',
    layout      = 'wide',
    initial_sidebar_state = 'expanded'
)

HF_TOKEN = os.environ.get('HF_TOKEN', None)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id   = 'Suhani2128/tourism-prediction-model',
        filename  = 'best_model.pkl',
        token     = HF_TOKEN,
        repo_type = 'model'
    )
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title('✈️ Wellness Tourism Package Predictor')
st.markdown('Predict whether a customer will purchase the **Wellness Tourism Package**.')
st.divider()

with st.sidebar:
    st.header('ℹ️ Model Info')
    st.info('**Algorithm:** XGBoost Classifier')
    st.info('**Tuning:** GridSearchCV · 3-fold CV')
    st.info('**Tracking:** MLflow')
    st.caption('Deployed via GitHub Actions | Hosted on HF Spaces')

st.subheader('👤 Customer Demographics')
col1, col2 = st.columns(2)

with col1:
    age            = st.slider('Age', 18, 65, 35)
    city_tier      = st.selectbox('City Tier', [1, 2, 3])
    # Gender: LabelEncoder order is 0=Fe Male, 1=Female, 2=Male
    # Show only Female/Male to user; map to correct encoded values
    gender_display = st.selectbox('Gender', ['Female', 'Male'])
    gender         = 1 if gender_display == 'Female' else 2
    marital_status = st.selectbox('Marital Status', [0, 1, 2, 3],
                                  format_func=lambda x: ['Divorced','Married','Single','Unmarried'][x])
    occupation     = st.selectbox('Occupation', [0, 1, 2, 3],
                                  format_func=lambda x: ['Free Lancer','Large Business','Salaried','Small Business'][x])
    designation    = st.selectbox('Designation', [0, 1, 2, 3, 4],
                                  format_func=lambda x: ['AVP','Executive','Manager','Senior Manager','VP'][x])
    monthly_income = st.number_input('Monthly Income (₹)', 1000, 100000, 22000, 500)

with col2:
    type_of_contact = st.selectbox('Type of Contact', [0, 1],
                                   format_func=lambda x: ['Company Invited','Self Enquiry'][x])
    product_pitched = st.selectbox('Product Pitched', [0, 1, 2, 3, 4],
                                   format_func=lambda x: ['Basic','Deluxe','King','Standard','Super Deluxe'][x])
    n_persons       = st.slider('Persons Visiting', 1, 5, 2)
    n_children      = st.slider('Children (<5 yrs)', 0, 3, 0)
    n_trips         = st.slider('Avg Annual Trips', 1, 22, 3)
    preferred_star  = st.slider('Preferred Property Star', 1, 5, 3)
    passport        = st.radio('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    own_car         = st.radio('Owns a Car?', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

st.divider()
st.subheader('📞 Sales Pitch Details')
col3, col4 = st.columns(2)
with col3:
    pitch_score = st.slider('Pitch Satisfaction Score', 1, 5, 3)
    n_followups = st.slider('Number of Follow-ups', 1, 6, 3)
with col4:
    duration = st.slider('Duration of Pitch (mins)', 5, 127, 30)

st.divider()

if st.button('🔮 Predict Purchase Likelihood', use_container_width=True, type='primary'):
    input_df = pd.DataFrame([{
        'Age'                     : age,
        'TypeofContact'           : type_of_contact,
        'CityTier'                : city_tier,
        'DurationOfPitch'         : duration,
        'Occupation'              : occupation,
        'Gender'                  : gender,
        'NumberOfPersonVisiting'  : n_persons,
        'NumberOfFollowups'       : n_followups,
        'ProductPitched'          : product_pitched,
        'PreferredPropertyStar'   : preferred_star,
        'MaritalStatus'           : marital_status,
        'NumberOfTrips'           : n_trips,
        'Passport'                : passport,
        'PitchSatisfactionScore'  : pitch_score,
        'OwnCar'                  : own_car,
        'NumberOfChildrenVisiting': n_children,
        'Designation'             : designation,
        'MonthlyIncome'           : monthly_income,
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader('📈 Prediction Result')
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1:
            st.success('✅ **LIKELY to Purchase** the Wellness Tourism Package')
            st.markdown('**Recommendation:** Prioritise this customer for outreach.')
        else:
            st.warning('❌ **UNLIKELY to Purchase** the Wellness Tourism Package')
            st.markdown('**Recommendation:** Consider alternative packages or defer.')
    with res_col2:
        st.metric('Purchase Probability', f'{probability * 100:.1f}%')
        st.progress(float(probability))

    with st.expander('🔍 View Input Data Submitted'):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

st.caption('© Visit with Us | Wellness Tourism Package Prediction | MLOps Pipeline')
