from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)
CSV_FILE_PATH = 'users.csv'
MODEL_FILE_PATH = 'trained_model.sav'

# Load and preprocess data
data = pd.read_csv("C:\\Users\\MEGHANA M\\documents\\DatasetFraud.csv")
data["type"] = data["type"].map({
    "CASH_OUT": 1, 
    "PAYMENT": 2,                              
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
})
data["isFraud"] = data["isFraud"].map({
    0: "No Fraud", 
    1: "Fraud"
})
X = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data["isFraud"])
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)

# Load the trained model
with open(MODEL_FILE_PATH, 'rb') as model_file:
    model = pickle.load(model_file)
def preprocess_and_predict(transaction):
    type_mapping = {
        "CASH_OUT": 1, 
        "PAYMENT": 2,                              
        "CASH_IN": 3,
        "TRANSFER": 4,
        "DEBIT": 5
    }
    transaction_type = type_mapping[transaction["type"]]
    amount = float(transaction["amount"])  # Ensure amount is float
    oldbalanceOrg = float(transaction["oldbalanceOrg"])  # Ensure balance fields are float
    newbalanceOrig = float(transaction["newbalanceOrig"])
    
    input_data = np.array([[transaction_type, amount, oldbalanceOrg, newbalanceOrig]])

    prediction = model.predict(input_data)
    prediction_label = "Fraud" if prediction[0] == "Fraud" else "No Fraud"
    
    return prediction_label
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/regSuccess', methods=['POST'])
def regSuccess():
    user = request.form.get('user')
    aadhar = request.form.get('aadhar')
    dob = request.form.get('dob')
    bank = request.form.get('bank')
    mobile_no = request.form.get('Mobile_no')
    credit_card = request.form.get('credit_card')

    # Create a dictionary for the new user
    new_user = {
        'Username': user,
        'Aadhar': aadhar,
        'DateOfBirth': dob,
        'Bank': bank,
        'MobileNumber': mobile_no,
        'CreditCard': credit_card,
        'balance': 10000,
        'TransactionTimes': "[]",
        'IPAddress': []  # Initialize an empty list for IP addresses
    }

    # Check if the CSV file exists
    if os.path.exists(CSV_FILE_PATH):
        # Load existing data
        users_df = pd.read_csv(CSV_FILE_PATH)
    else:
        # Create a new DataFrame if the file does not exist
        users_df = pd.DataFrame(columns=['Username', 'Aadhar', 'DateOfBirth', 'Bank', 'MobileNumber', 'CreditCard', 'balance', 'TransactionTimes', 'IPAddress'])

    # Append the new user to the DataFrame using pd.concat
    new_user_df = pd.DataFrame([new_user])
    users_df = pd.concat([users_df, new_user_df], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    users_df.to_csv(CSV_FILE_PATH, index=False)

    return render_template('home.html')

@app.route('/transaction')
def transaction():
    return render_template("transaction.html")

@app.route('/validate', methods=['POST'])
def validate():
    try:
        # Log the received form data
        logging.info(f"Form data received from IP address: {request.remote_addr}, Data: {request.form}")

        transaction = {
            "type": request.form["tp"],
            "amount": float(request.form["TransactionAmt"]),
            "credit_card": int(request.form["credit_card"])
        }

        # Load existing data
        users_df = pd.read_csv(CSV_FILE_PATH)

        # Check if the credit card number exists in the DataFrame
        if transaction["credit_card"] in users_df['CreditCard'].values:
            # Find the user's record using the credit card number
            user_index = users_df[users_df['CreditCard'] == transaction["credit_card"]].index[0]

            # Check if there are previous transactions
            transaction_times = eval(users_df.at[user_index, 'TransactionTimes'])
            if transaction_times:
                last_transaction_time = datetime.fromisoformat(transaction_times[-1])
                current_time = datetime.now()
                time_diff = (current_time - last_transaction_time).total_seconds()
                if time_diff < 60:
                    # Cancel the transaction if the time difference is less than 1 minute
                    msg = "Transaction canceled due to frequent transaction attempts. Please try again later."
                    return redirect(url_for('paymentStatus', msg=msg))

            # Update the TransactionTimes
            current_time = datetime.now().isoformat()
            transaction_times.append(current_time)
            users_df.at[user_index, 'TransactionTimes'] = str(transaction_times)

            # Update old balance and new balance
            old_balance = float(users_df.at[user_index, 'balance'])
            new_balance = old_balance - transaction["amount"]
            users_df.at[user_index, 'balance'] = new_balance

            # Append IP address to the list
            ip_addresses = eval(users_df.at[user_index, 'IPAddress'])
            ip_addresses.append(request.remote_addr)
            users_df.at[user_index, 'IPAddress'] = str(ip_addresses)

            # Save the updated DataFrame to the CSV file
            users_df.to_csv(CSV_FILE_PATH, index=False)

            # Predict fraud based on transaction details
            transaction["oldbalanceOrg"] = old_balance
            transaction["newbalanceOrig"] = new_balance
            msg = preprocess_and_predict(transaction)

        else:
            msg = "Credit card number not found"

    except ValueError:
        msg = "Invalid transaction"
    except Exception as e:
        msg = f"Error in processing: {str(e)}"
        logging.error(f"Error in processing: {str(e)}")

    return redirect(url_for('paymentStatus', msg=msg))

@app.route('/paymentStatus')
def paymentStatus():
    msg = request.args.get('msg', 'No message provided')
    return render_template("paymentStatus.html", msg=msg)

if __name__ == "__main__":
    app.run(debug=True)
