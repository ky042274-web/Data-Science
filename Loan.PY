import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Set Background Image
# -----------------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://static.vecteezy.com/system/resources/previews/024/269/311/non_2x/car-house-personal-money-loan-concept-finance-business-icon-on-wooden-cube-saving-money-for-a-car-money-and-house-wooden-cubes-with-word-loan-copy-space-for-text-loan-payment-car-and-house-photo.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.7);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -----------------------------
# Loan Approval App
# -----------------------------
st.title("🏠🚗 Loan Approval Prediction App")

st.write("This app predicts loan approval using Logistic Regression. The background represents the loan and finance theme you selected.")

# Example dataset
data = {
    "Age":[33,21,57,50,21,43,30,23,41,26,58,31,42,37,49,59,60,41,32,39,40,26,23,35,53,41,41,27,31,30,60,35,58,31,33,38,39,38,40,52,33,50,25,51,59,36,51,35,25,28,25,57,41,31,55,44,22,38,30,33],
    "Income":[95457,21105,88970,92604,50622,37833,62531,72585,44272,29858,74880,53721,94701,21110,52851,64501,43637,22641,24905,29593,60450,77906,65468,37819,33335,98929,55311,25589,52488,85963,47272,71009,25736,69066,61361,77537,61786,58906,23342,39360,38812,90134,97000,54265,67261,82621,24476,25625,53113,36116,84221,72040,65989,74237,83823,75670,91333,64217,72309,58599],
    "CreditScore":[349,650,569,709,305,575,508,614,536,428,593,717,684,706,788,720,365,491,456,647,514,717,456,460,802,714,312,499,671,317,444,546,418,415,356,691,479,760,590,569,402,771,606,360,300,648,376,385,688,577,846,384,303,722,732,637,418,751,579,672],
    "LoanApproved":[0,0,0,1,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,1,0,1]
}
df = pd.DataFrame(data)

X = df[["Age","Income","CreditScore"]]
y = df["LoanApproved"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

# -----------------------------
# Show Accuracy
# -----------------------------
st.subheader("Model Accuracy")
st.write(f"Accuracy on test data: {model.score(x_test, y_test):.2f}")

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Applicant Details")
age = st.number_input("Age", min_value=18, max_value=70, value=30)
income = st.number_input("Income", min_value=10000, max_value=100000, value=50000)
credit = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

if st.button("Predict Loan Approval"):
    new_data = pd.DataFrame([[age, income, credit]], columns=["Age","Income","CreditScore"])
    prediction = model.predict(new_data)[0]
    st.success("Loan Approval: YES ✅" if prediction==1 else "Loan Approval: NO ❌")
