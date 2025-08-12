import pickle
from fastapi import FastAPI, UploadFile, File
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load the trained model from the pickle file
with open('best_model.pkl', 'rb') as model:
    model = pickle.load(model)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler_file = pickle.load(scaler_file)

app = FastAPI()
@app.get("/")
def health_check():
    return {"health check": "Ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #read data
    content = await file.read()
    data = pd.read_csv(io.BytesIO(content))
    #make sure the columns are the same as the training data
    expected_columns = ['visitorid', 'ses_rec', 'ses_rec_avg', 'ses_rec_sd', 'ses_rec_cv',
       'user_rec', 'ses_n', 'ses_n_r', 'int_n', 'int_n_r', 'tran_n',
       'tran_n_r', 'rev_sum', 'rev_sum_r', 'major_spend_r', 'int_cat_n_avg',
       'int_itm_n_avg', 'ses_mo_avg', 'ses_mo_sd', 'ses_ho_avg', 'ses_ho_sd',
       'ses_wknd_r', 'ses_len_avg', 'time_to_int', 'time_to_tran',
       'int_cat1_n', 'int_cat2_n', 'int_cat3_n', 'int_cat4_n', 'int_cat5_n',
       'int_cat6_n', 'int_cat7_n', 'int_cat8_n', 'int_cat9_n', 'int_cat10_n',
       'int_cat11_n', 'int_cat12_n', 'int_cat13_n', 'int_cat15_n',
       'int_cat16_n', 'int_cat17_n', 'int_cat18_n', 'int_cat19_n',
       'int_cat20_n', 'int_cat21_n', 'int_cat22_n', 'int_cat23_n',
       'int_cat24_n']

    if list(data.columns) != expected_columns:
        return {"error": "Input data does not match the expected format."}
    
    #prep data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # Make predictions using the loaded model
    predictions = model.predict(data_scaled)
    return {"predictions": predictions.tolist()}
