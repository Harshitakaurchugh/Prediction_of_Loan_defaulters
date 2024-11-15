from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
with open('gb_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    input_features = [
        float(request.form['Age']),
        float(request.form['Income']),
        float(request.form['LoanAmount']),
        float(request.form['CreditScore']),
        float(request.form['MonthsEmployed']),
        float(request.form['NumCreditLines']),
        float(request.form['InterestRate']),
        float(request.form['LoanTerm']),
        float(request.form['DTIRatio']),
        float(request.form['HasMortgage']),
        float(request.form['HasDependents']),
        float(request.form['HasCoSigner']),
        float(request.form['LoanToIncomeRatio']),
        float(request.form['CreditUtilizationRate']),
        float(request.form['Education_HighSchool']),
        float(request.form['Education_Masters']),
        float(request.form['Education_PhD']),
        float(request.form['EmploymentType_PartTime']),
        float(request.form['EmploymentType_SelfEmployed']),
        float(request.form['EmploymentType_Unemployed']),
        float(request.form['MaritalStatus_Married']),
        float(request.form['MaritalStatus_Single']),
        float(request.form['LoanPurpose_Business']),
        float(request.form['LoanPurpose_Education']),
        float(request.form['LoanPurpose_Home']),
        float(request.form['LoanPurpose_Other'])
    ]

    # Predict using the loaded model
    prediction = model.predict([input_features])

    # Display the result on the web page
    result = "Default" if prediction[0] == 1 else "No Default"
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)

