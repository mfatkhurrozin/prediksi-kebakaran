#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# import the data
data = pd.read_csv("smoke_detection_iot_clean.csv")
image = Image.open("api.png")
st.title("Selamat datang di Prediksi Kebakaran")
st.image(image, use_column_width=True)

# checking the data
st.write("Aplikasi ini bertujuan unutuk memprediksi terjadi kebakaran atau tidaknya")
check_data = st.checkbox("Lihat contoh data")
if check_data:
    st.write(data[1:10])
st.write("Sekarang masukan data parameter untuk melihat hasil prediksi")

# input the numbers
temp = st.slider("Temperatur(C)", int(
    data.temperature.min()), int(data.temperature.max()), int(data.temperature.mean()))
humid = st.slider("Kelembapan Udara", int(
    data.humidity.min()), int(data.humidity.max()), int(data.humidity.mean()))
tv = st.slider("Total Senyawa Organik Volatil(ppb)", int(data.tvoc.min()), int(
    data.tvoc.max()), int(data.tvoc.mean()))
ecodua = st.slider("CO2 equivalent concentration(ppm)", int(data.ecodua.min()), int(
    data.ecodua.max()), int(data.ecodua.mean()))
rawhdua = st.slider("Jumlah Hidrogen Mentah yang ada di sekitarnya", int(data.rawhdua.min()), int(
    data.rawhdua.max()), int(data.rawhdua.mean()))
rawethanol = st.slider("Jumlah Etanol Mentah yang ada di sekitarnya", int(data.rawethanol.min()), int(
    data.rawethanol.max()), int(data.rawethanol.mean()))
pressure = st.slider("Tekanan Udara", int(data.pressure.min()), int(
    data.pressure.max()), int(data.pressure.mean()))
nckosonglima = st.slider("Konsentrasi partikel dengan diameter kurang dari 0,5 mikrometer", int(data.nckosonglima.min()), int(
    data.nckosonglima.max()), int(data.nckosonglima.mean()))
ncsatu = st.slider("Konsentrasi partikel dengan diameter kurang dari 1,0 mikrometer", int(data.ncsatu.min()), int(
    data.ncsatu.max()), int(data.ncsatu.mean()))
# splitting your data
X = data.drop('firealarm', axis=1)
y = data['firealarm']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=45)

# # modelling step
# # Linear Regression model
# # import your model
# model = LinearRegression()
# # fitting and predict your model
# model.fit(X_train, y_train)
# model.predict(X_test)
# errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
# predictions = model.predict(
#     [[temp, humid, tv, ecodua, rawhdua, rawethanol]])[0]
# akurasi = np.sqrt(r2_score(y_test, model.predict(X_test)))

# =============================================================================
# #RandomForestModel
#model2 = RandomForestRegressor(random_state=0)
# model2.fit(X_train,y_train)
# model2.predict(X_test)
#errors = np.sqrt(mean_squared_error(y_test,model2.predict(X_test)))
#predictions = model2.predict([[temp,humid,tv,ecodua,rawhdua,rawethanol]])[0]
#akurasi= np.sqrt(r2_score(y_test,model2.predict(X_test)))
# =============================================================================

# =============================================================================
# DecissionTreeModel
model3 = DecisionTreeRegressor(random_state=42)
model3.fit(X_train, y_train)
model3.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test, model3.predict(X_test)))
predictions = model3.predict(
    [[temp, humid, tv, ecodua, rawhdua, rawethanol, pressure, nckosonglima, ncsatu]])[0]
akurasi = np.sqrt(r2_score(y_test, model3.predict(X_test)))
# =============================================================================

if predictions == 1:
    hasil = ("Terjadi Kebakaran")
else:
    hasil = ("Tidak Terjadi Kebakaran")

hasilakurasi = akurasi*100
# checking prediction house price
if st.button("Prediksi!"):
    st.header("Hasil Prediksi : {}".format(hasil))
    st.subheader("Tingkat Akurasi : {}".format(hasilakurasi))
