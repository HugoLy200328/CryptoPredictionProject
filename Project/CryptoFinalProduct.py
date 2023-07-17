import datetime as dt
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import xgboost as xgb
import seaborn as sns
import sklearn

# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split


# Set the page title
st.set_page_config(page_title="PROJECT: CRYPTOCURRENCY PREDICTION USING MACHINE LEARNING",
                   page_icon=":money_with_wings:")

# Set sidebar
# st.title('PROJECT: CRYPTOCURRENCY PREDICTION USING MACHINE LEARNING')
# Replace with the actual path to your image
image_path = "./cryptocurrency-3d-illustration-free-png.png"
st.sidebar.image(image_path, caption="Image in Sidebar", use_column_width=True)
st.sidebar.title('`DASHBOARD`')
st.sidebar.markdown('Lecture: **_Do Duy Thanh_**')
st.sidebar.write(' The Team:\n'
                 ' - Ly Gia Hieu\n'
                 ' - Nguyen Xuan Yen\n'
                 ' - Tran Ngoc Nhu Hao\n')


# Declare global variable
eth_data = None

# Define function to generate ETH dataset


def train_test_split(X, y, test_size=0.2, random_state=None):
    X_shuffled, y_shuffled = X, y
    split_index = int(len(X) * (1 - test_size))
    X_train = X_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]
    return X_train, X_test, y_train, y_test


def generate_eth_dataset():
    # Set the start and end dates for the data
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.date.today()
    # Retrieve ETH data
    eth_data = yf.download('ETH-USD', start=start_date, end=end_date)
    # Display the ETH dataset
    return eth_data


def generate_eth_dataset_last(range):
    # Set the start and end dates for the data
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=range)
    # Retrieve ETH data
    eth_data = yf.download('ETH-USD', start=start_date, end=end_date)
    # Display the ETH dataset
    return eth_data

# Define function to process the dataset


def process_dataset(eth_data):
    # Resample the DataFrame to fill any missing dates
    eth_data = eth_data.resample('D').ffill()

    # Create X and y
    X = eth_data.drop('Close', axis=1)
    y = eth_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    # Creating XGBoost train and test data
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dtest = xgb.DMatrix(X_test.values, label=y_test.values)

    return eth_data, X, y, X_train, X_test, y_train, y_test, dtrain, dtest

# Define function to train the XGBoost model


def train_xgboost_model(dtrain, num_round=1000):
    # Training modelll ============================== START
    # Defining hyperparameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 9,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'verbosity': 0
    }
    # Training the XGBoost model
    xgb_model = xgb.train(params, dtrain, num_round)
    # Training modelll ============================== END
    return xgb_model


# Define function to predict on the testing dataset
def predict_on_testing_dataset(xgb_model, dtest, y_test):
    # Perform prediction
    prediction = xgb_model.predict(dtest)
    # Convert dataframe to an array for plotting graphs purposes
    y_test_transformed = y_test.values
    return prediction, y_test_transformed

# Predict days


def predict_lastest_days(xgb_model, dtest, y_test):
    # Perform prediction
    prediction = xgb_model.predict(dtest)
    # Convert dataframe to an array for plotting graphs purposes
    y_test_transformed = y_test.values
    return prediction, y_test_transformed


# ===============
# Define function to create navigation bar
crypto_list_select = ["BNB-USD", "ETH-USD", "XRP-USD", "BTC-USD", "ADA-USD"]


def create_navbar():
    global eth_data
    menu = ["The project", "Crypto currency price",
            "Compare Crypto currency price", "Run the project", "Predict the lastest days"]
    choice = st.sidebar.selectbox("Navigation", menu)
    if choice == "The project":
        st.markdown(
            f"""
            <h1 style='width: 800px; text-align: start;'>PROJECT: CRYPTOCURRENCY PREDICTION USING MACHINE LEARNING</h1>
            """,
            unsafe_allow_html=True)
        # ---- ABSTRACT!!!
        st.header('Abstract')
        st.markdown('<div align="justify">Ethereum has a significant impact on blockchain technology. In the project, proposed to correctly forecast the Ethereum price while taking into account a number of factors that influence the Ethereum value. In addition to learning about the best features related to Ethereum price, our goal is to comprehend and identify everyday trends in the Ethereum market. The data set comprises different elements that have been tracked daily over the course of each year in relation to the Ethereum price and payment network. To forecast the closing price of the following day, factors including the opening price, highest price, lowest price, closing price, volume of Ethereum. Using the Scikit-Learn tools, Keras tools.</div>', unsafe_allow_html=True)
        # ---- INTRODUCTION!!!
        # I. Introduction
        st.header('I. Introduction')
        st.markdown('<div align="justify">Ethereum, a pioneering blockchain platform, has captivated the world with its potential to revolutionize decentralized applications and smart contracts. As the popularity and adoption of Ethereum continue to soar, the question of predicting its future price has become a matter of great interest for investors, traders, and researchers alike. Machine learning algorithms have emerged as a promising avenue for forecasting the price of Ethereum.Machine learning models have the ability to analyze vast amounts of data, including historical price data, market sentiment, network activity, and fundamental factors, to uncover patterns and make predictions. These models employ various techniques, such as regression, time series analysis, and deep learning, to understand and predict the future price of Ethereum.However, it is crucial to acknowledge that predicting the price of Ethereum, or any cryptocurrency for that matter, is a formidable challenge due to the highly volatile and unpredictable nature of the cryptocurrency market. Factors such as regulatory changes, technological advancements, market sentiment, and global economic events can significantly impact the price of Ethereum, making accurate predictions a complex endeavor.Despite the challenges, machine learning holds tremendous promise for Ethereum price prediction. By harnessing the power of advanced algorithms, machine learning models can potentially identify hidden patterns and relationships within the data that may elude human analysts. These models have the potential to provide valuable insights for investors and traders, enabling them to make informed decisions regarding buying, selling, and holding Ethereum.It is worth noting that any predictions generated by machine learning models should be interpreted with caution and not be solely relied upon as investment advice. The cryptocurrency market is highly dynamic, and unexpected events or sudden shifts in market sentiment can lead to rapid and substantial price fluctuations.As the field of machine learning continues to evolve and new advancements are made, it is likely that more sophisticated and accurate models will be developed for Ethereum price prediction. Researchers and data scientists are continually exploring novel techniques and incorporating additional data sources to enhance the predictive capabilities of these models.In conclusion, machine learning offers a promising approach to predict the future price of Ethereum. While it holds the potential to assist investors and traders in making informed decisions, it is essential to remain cognizant of the challenges inherent in forecasting cryptocurrency prices. As Ethereum continues to shape the decentralized landscape, the combination of machine learning and Ethereum price prediction holds exciting prospects for the future.</div>', unsafe_allow_html=True)
        # ---- MODEL
        st.markdown(
            f"""
            <h3 style='align="justify;'>1. Model</h3>
            """,
            unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.1. MLP</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Multilayer Perceptron (MLP) is a popular and effective machine learning approach used for predicting cryptocurrency prices. MLP is a type of artificial neural network that can analyze complex patterns and relationships within cryptocurrency data, enabling accurate predictions of future price movements. By adjusting weights based on historical data, MLP can learn from the past to forecast cryptocurrency prices.  MLP\'s ability to capture non-linear relationships and handle high volatility makes it valuable for cryptocurrency price prediction. It considers various input features, such as historical prices, trading volumes, market sentiment, and technical indicators, to make predictions beyond the capabilities of linear models. However, challenges such as market volatility and hyperparameter sensitivity should be considered.</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <h4 style='align="justify;'>1.2. CRU</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Gated Recurrent Unit (GRU) is a powerful type of recurrent neural network (RNN) architecture used for cryptocurrency price prediction. GRU models excel at capturing temporal dependencies and analysing sequential data. By incorporating gating mechanisms, GRU models selectively update and forget information, allowing them to capture long-term patterns in cryptocurrency price data. GRU models are flexible and adaptable, making them well-suited for the unpredictable and volatile nature of cryptocurrency markets. They can process historical price data, trading volumes, and other relevant factors to make accurate predictions about future price movements. However, achieving reliable predictions requires careful consideration of hyperparameters and training data.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.3. CNN</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Convolutional Neural Network (CNN) is a popular deep learning architecture widely used for image analysis. In cryptocurrency price prediction, CNN models have shown promise in capturing patterns and extracting features from time-series data. CNN models excel at processing structured data with grid-like properties. When applied to cryptocurrency price prediction, CNN models can analyse historical price sequences and identify significant patterns or trends. By convolving filters over the data, they capture local dependencies and extract informative features. The ability of CNN models to capture short-term and long-term dependencies makes them suitable for forecasting in the volatile cryptocurrency market. While challenges remain due to market dynamics and external factors, ongoing research aims to improve CNN architectures for more accurate predictions.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.4. LSTM</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">LSTM (Long Short-Term Memory) is a specialized recurrent neural network architecture known for its ability to capture long-term dependencies in sequential data. In cryptocurrency price prediction, LSTM models have gained popularity due to their effectiveness in capturing complex temporal relationships. LSTM models utilize memory cells and gating mechanisms to selectively retain or forget information over time, enabling them to learn patterns from historical cryptocurrency price data and make accurate predictions about future price movements.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.5. BI-LSTM</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Bidirectional Long Short-Term Memory (BI-LSTM) is an advanced variant of LSTM, specialized in capturing both past and future dependencies in sequential data. In cryptocurrency price prediction, BI-LSTM models offer improved capabilities for analyzing historical price sequences and making accurate forecasts. BI-LSTM models process data in both forward and backward directions, capturing comprehensive temporal dynamics in cryptocurrency prices. By considering past and future information simultaneously, they excel at capturing complex patterns and dependencies within the data.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.6. XGBoost</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">XGBoost is a powerful machine learning algorithm used for precise predictions, including in cryptocurrency price forecasting. It combines weak models to create a robust predictor and handles complex patterns effectively. In cryptocurrency price prediction, XGBoost analyzes features like historical prices, market indicators, and sentiment analysis. By identifying patterns and relationships, it provides accurate forecasts of future price movements. XGBoost excels in handling high-dimensional data, dealing with missing values and outliers, and preventing overfitting through regularization techniques.Although predicting cryptocurrency prices is challenging, XGBoost offers valuable insights for investors and traders, assisting in informed decision-making.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.7. Random Forest</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Random Forest is a versatile machine learning algorithm known for accurate predictions. In cryptocurrency price prediction, Random Forest has gained popularity for handling complex patterns effectively. Random Forest combines multiple decision trees to create a powerful predictor. By analyzing features like historical prices, market indicators, and sentiment analysis, it provides accurate forecasts of future price movements. The algorithm excels in handling high-dimensional data, non-linear relationships, and missing values. It is less prone to overfitting and offers insights into feature importance. While predicting cryptocurrency prices is challenging, Random Forest provides valuable insights for investors and traders, aiding decision-making in the cryptocurrency market.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>1.8. SVR</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Support Vector Regression (SVR) is a popular machine learning algorithm used for predicting continuous values, making it suitable for cryptocurrency price prediction. SVR finds a regression function by utilizing support vectors and creating a hyperplane that best fits the data. In cryptocurrency price prediction, SVR analyzes various features such as historical price data, trading volumes, market indicators, and sentiment analysis. By capturing complex patterns and using kernel functions, SVR models can make accurate predictions about future cryptocurrency prices.</div>', unsafe_allow_html=True)
        # ---- EVALUATION CRITERIA
        st.markdown(
            f"""
            <h3 style='align="justify;'>2. Evaluation criteria</h3>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) are commonly used metrics for evaluating the performance of prediction models. These metrics provide insights into the accuracy and precision of predictions in various domains, including cryptocurrency price prediction.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.1 MSE</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Measures the average squared difference between the predicted and actual values. It calculates the variability of errors and penalizes larger deviations more significantly. A lower MSE value indicates better prediction accuracy, with zero indicating a perfect prediction.</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Evaluation Criteria/MSE.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.2 RMSE</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">Measures the average squared difference between the predicted and actual values. It calculates the variability of errors and penalizes larger deviations more significantly. A lower MSE value indicates better prediction accuracy, with zero indicating a perfect prediction.</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Evaluation Criteria/RMSE.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.3 MAE</h4>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">MAE on the other hand, calculates the average absolute difference between the predicted and actual values. It measures the average magnitude of errors, regardless of their direction. Like MSE and RMSE, a lower MAE value signifies better prediction performance.</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Evaluation Criteria/MAE.png')
        st.markdown('<div align="justify">These evaluation metrics play a crucial role in assessing the effectiveness of prediction models in cryptocurrency price forecasting. They provide quantitative measures to compare different models, select the best-performing one, and assess their reliability in real-world scenarios.</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h3 style='align="justify;'>3. Dataset</h3>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify">The dataset we are using in this project is the ETH USDT daily dataset from CryptoDataDownload (CDD) - a website that provides historical cryptocurrency data and trading tools for cryptocurrency enthusiasts and traders. It offers a wide range of data, including price and volume data, futures data, options data, order book snapshots, and more. The website aims to empower traders and researchers by providing them with comprehensive and reliable data for their analysis and strategies. Get data from trustworthy Crypto current Exchange from all over the world like Gemini, Kraken, Binance,...</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Dataset/Corral.png')
        st.markdown('<div align="justify">Based on the correlation matrix, we can see the correlation between features in the dataset and help choose the best features for building the models. </div>', unsafe_allow_html=True)
        st.markdown('<div align="justify">The ETH daily dataset from CryptoDataDownload provides historical daily price data for Ethereum (ETH), allowing analysis and understanding the price movements of ETH over time. Here are some main features of the dataset we are gonna use in the project:</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify"> -    Opening: The price of ETH at the beginning of the day. </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify"> -    Closing: The price of ETH at the end of the day. High: The highest price of ETH reached during the day.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify"> -    Low: The lowest price of ETH reached during the day. </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify"> -    Volume ETH: Volume represents the number of ETH coins traded during a specific day. </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify"> -    Volume USDT: Volume represents the number of USDT traded during a specific day. </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Dataset/InText.png')
        st.markdown(
            '<div align="justify"> We will be predicting the closing price of the Ethereum (ETH) of each day so to have a better visualization and understanding of the dataset, this is the Line Graph of the Close feature in the dataset: </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Dataset/Line.png')
        # II. Methodology
        st.header('II. Methodology')
        st.markdown(
            '<div align="justify"> The flow diagram:', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/flowDia.png')
        st.markdown(
            f"""
            <h3 style='align="justify;'>1. Pre-processing dataset</h3>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify"> The dataset we got from CryptoDataDownload was misorganized, so we have to sort the entire dataset to fit the index from the oldest to the latest dates. </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/Read.png')
        st.image('./PicForProduct/Methodology/Pre-Processing/Sort.png')
        st.markdown('<div align="justify">While working on the project, we encountered missing values for a certain day in the dataset. But it was only a really small percentage of the dataset so for dealing with this problem we are using the Fill function in the pandas library that can be used to forward fill missing values in a time series data when resampling the data to a different frequency.</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/fILL.png')
        st.markdown('<div align="justify">We drop all the unnecessary features in the dataset (features that do not play or play a really small role in the prediction of the closing price)</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/Drop.png')
        st.markdown('<div align="justify">We build little data frames consisting of 10 consecutive days of data called windows</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/Scalling12.png')

        st.markdown(
            '<div align="justify">Scaling data into a specific range:</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/Scaling3.png')
        st.image('./PicForProduct/Methodology/Pre-Processing/Scaling4.png')
        st.markdown('<div align="justify">Split the data into two sets — training set and test set with 80% and 20% data respectively</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/TT.png')
        st.markdown('<div align="justify">While building the XGB model, the train and test dataset was not match so we dealt with it by creating data matrices that can be used for training and testing the XGBoost model.</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Pre-Processing/forXGB.png')
        st.markdown(
            f"""
            <h3 style='align="justify;'>2. Build and Train models</h3>
            """,
            unsafe_allow_html=True)
        st.markdown('<div align="justify" >There are a total of 8 machine learning models we are using in the project, the main idea was to find the most efficient models. </div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.1. MLP</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/MLP/code.png')
        st.image('./PicForProduct/Methodology/Build/MLP/loss.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.2. CRU</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/GRU/code.png')
        st.image('./PicForProduct/Methodology/Build/GRU/loss.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.3. CNN</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/CNN/code.png')
        st.image('./PicForProduct/Methodology/Build/CNN/loss.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.4. LSTM</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/LSTM/code.png')
        st.image('./PicForProduct/Methodology/Build/MLP/loss.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.5. BI-LSTM</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/BILSTM/code.png')
        st.image('./PicForProduct/Methodology/Build/BILSTM/loss.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.6. XGBoost</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/XGB/code.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.7. Random Forest</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/RF/code.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>2.8. SVR</h4>
            """,
            unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Build/SVR/code.png')

        st.markdown(
            f"""
            <h3 style='align="justify;'>3. Testing models</h3>
            """,
            unsafe_allow_html=True)

        st.markdown(
            f"""
            <h4 style='align="justify;'>3.1. MLP</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >MLP MSE: 0.0006500998699703938 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >MLP MAE: 0.02047678372714629 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >MLP RMSE: 0.025497056103997454 </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/MLP/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.2. CRU</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >GRU MSE: 0.0005600607069121045 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >GRU MAE: 0.016615089026171384 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >GRU RMSE: 0.0236656017652648 </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/GRU/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.3. CNN</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >CNN MSE: 0.0042914414891870555 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >CNN MAE: 0.05698704394851298 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >CNN RMSE: 0.06550909470590366 </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/CNN/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.4. LSTM</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >LSTM MSE: 0.0008363100195527774 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >LSTM MAE: 0.020494522704895775 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >LSTM RMSE: 0.028919025217886883 </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/LSTM/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.5. BI-LSTM</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >BI-LSTM MSE: 0.0009980063750317978 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >BI-LSTM MAE: 0.022464854164516372</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >BI-LSTM RMSE: 0.031591238896754234</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/BILSTM/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.6. XGBoost</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >XGBoost MSE: 0.0004218476902704535 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >XGBoost MAE: 0.015352940836973547 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >XGBoost RMSE: 0.020538931088799475 </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/XGB/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.7. Random Forest</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >Random Forest MSE: 0.00043639364418259597 </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >Random Forest MAE: 0.01569254988330534</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >Random Forest RMSE: 0.020890036959818812</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/RF/line.png')
        st.markdown(
            f"""
            <h4 style='align="justify;'>3.8. SVR</h4>
            """,
            unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >SVR MSE: 0.0004337785901524045</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >SVR MAE: 0.016467762945042028</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify" >SVR RMSE: 0.020827351971683884 </div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Methodology/Testing/SVR/line.png')
        # IV. Result and observations
        st.header('III. Result ')
        st.markdown(
            '<div align="justify">Comparing the prediction of every models and the actual price:</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Result/line.png')
        st.markdown(
            '<div align="justify">Comparing the MSE of every models:</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Result/purple.png')
        st.markdown(
            '<div align="justify">Comparing the RMSE of every models:</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Result/green.png')
        st.markdown(
            '<div align="justify">Comparing the MAE of every models:</div>', unsafe_allow_html=True)
        st.image('./PicForProduct/Result/orange.png')
        st.markdown(
            '<div align="justify">Performance Comparison: Among the models evaluated, the XGBoost Model appears to have the lowest MSE (0.0004218476902704535), indicating that it has the smallest average squared difference between predicted and actual values. The Gated Recurrent Unit (GRU) model follows closely with a slightly higher MSE (0.0005600607069121045), followed by the Random Forest Model (0.00043639364418259597) and the Support Vector Regression (SVR) Model (0.0004337785901524045).</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">Accuracy of Predictions: The MAE values provide insights into the average absolute difference between predicted and actual values. The XGBoost Model and the Random Forest Model show the lowest MAE values, indicating that, on average, their predictions have the smallest deviation from the actual values. The GRU model also performs well in terms of MAE.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">Model Complexity: The models evaluated encompass various architectures, including Multilayer Perceptron, Gated Recurrent Unit, Convolutional Neural Network, Long short-term memory (LSTM), Bi Long short-term memory (Bi-LSTM), XGBoost, Random Forest, and SVR models. The results indicate that the neural network-based models (GRU, LSTM, and Bi-LSTM) generally have higher MSE and RMSE values compared to the XGBoost and Random Forest models. This suggests that the latter models may provide better predictions for the specific task at hand.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">Overall Recommendation: Based on the provided evaluation metrics, the XGBoost Model stands out as the most accurate model, considering its lowest MSE, MAE, and RMSE values. It demonstrates superior predictive performance in comparison to the other models evaluated. The Random Forest Model also performs well and can be considered as an alternative choice. However, it is crucial to consider other factors, such as model training time, interpretability, and specific requirements of the task, when selecting the most suitable model for deployment.</div>', unsafe_allow_html=True)

        # V. Conclusion and future work
        # V. Conclusion and future work
        st.header('IV. Conclusion')
        st.markdown(
            '<div align="justify">Cryptocurrency price prediction using machine learning has gained significant attention in recent years as a promising research area. Various algorithms, including neural networks, random forests, and support vector regression, have been utilized to forecast cryptocurrency prices. These models incorporate historical price data, trading volumes, social media sentiment, and other relevant factors to make predictions about future prices. However, it is important to note that cryptocurrency markets are highly volatile and unpredictable, making the development of accurate price prediction models challenging. Even the best models can sometimes fail to provide accurate forecasts. Moreover, using machine learning predictions alone for cryptocurrency trading carries certain risks, and it is advisable to exercise caution. The field of cryptocurrency price prediction using machine learning is dynamic and continuously evolving. Several studies have been conducted on cryptocurrencies like Bitcoin, Ethereum, and Litecoin, employing algorithms such as neural networks, random forests, and support vector machines. In conclusion, while machine learning can be a valuable tool for predicting cryptocurrency prices, it should be complemented with other forms of analysis and approached with caution. Unforeseeable factors and market conditions can cause price fluctuations that cannot be accurately predicted by any model.</div>', unsafe_allow_html=True)
        # VI. References
        st.header('V. References')
        st.markdown(
            '<div align="justify">1.	PRICE PREDICTION OF DIFFERENT CRYPTOCURRENCIES USING TECHNICAL TRADE INDICATORS AND MACHINE LEARNING - Mohammed Salman and Abdullahi Abdu IBrahim.</div>', unsafe_allow_html=True)
        st.write(
            "(https://iopscience.iop.org/article/10.1088/1757-899X/928/3/032007/meta)")
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">2.	Machine learning for cryptocurrency market prediction and trading - Patrick Jaquart, Sven Köpke, Christof Weinhardt</div>', unsafe_allow_html=True)
        st.write(
            "(https://www.sciencedirect.com/science/article/pii/S2405918822000174)")
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">3.	The Legal Nature of Cryptocurrency - Bolotaeva, Stepanova and Alekseeva.</div>', unsafe_allow_html=True)
        st.write(
            "(https://iopscience.iop.org/article/10.1088/1755-1315/272/3/032166/meta)")
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">4.	Prediction of cryptocurrency returns using machine learning - Erdinc Akyildirim, Ahmet Goncu & Ahmet Sensoy</div>', unsafe_allow_html=True)
        st.write(
            "(https://www.researchgate.net/publication/329322600_Prediction_of_Cryptocurrency_Returns_using_Machine_Learning)")
        st.markdown(
            '<div align="justify">  </div>', unsafe_allow_html=True)
        st.markdown(
            '<div align="justify">5.	Time-series  Bitcoin  price predictions  with  high-dimensional  features  using machine  learningMohammed Mudassir1 Devrim Unal1 Shada Bennbaia 20 June Mohammad Hammoudeha division of Springer Nature 2020 is Springer-Verlag London Ltd.</div>', unsafe_allow_html=True)
        st.write(
            "(https://pubmed.ncbi.nlm.nih.gov/32836901/)")
        st.markdown(
            '<div align="justify">6.    Here are some youtube video link that i had studied for these project</div>', unsafe_allow_html=True)
        st.write(
            "[What is LSTM (Long Short Term Memory)?](https://www.youtube.com/watch?v=b61DPVFX03I)")
        st.write(
            "[Predicting Crypto Prices in Python](https://www.youtube.com/watch?v=GFSiL6zEZF0)")
        st.write(
            "[Time Series Forecasting with XGBoost - Use python and machine learning to predict energy consumption](https://www.youtube.com/watch?v=vV12dGe_Fho)")
        st.write(
            "[Predict The Future Price of Ethereum Using Python & Machine Learning](https://www.youtube.com/watch?v=OcpAkACOwW0&t=45s)")
        st.write(
            "[Ethereum (ETH) Price Prediction using Machine Learning (SVR) & Python](https://www.youtube.com/watch?v=HiDEAWdAif0)")

    elif choice == "Crypto currency price":
        # Set up Streamlit app title
        st.title("Cryptocurrency Price Chart")

        # Define list of available cryptocurrencies
        crypto_list = crypto_list_select

        # ===Display dropdown menu to select cryptocurrency
        selected_crypto = st.selectbox(
            "Select a cryptocurrency", crypto_list, key='slb1')

        # Define time range for historical price data
        time_range = st.selectbox("Select time range", [
            "1d", "5d", "1mo", "6mo", "1y", "5y"], key='slb2')

        # Get historical price data using yfinance library
        crypto_data = yf.download(selected_crypto, period=time_range)

        # Create line chart of historical price data using Plotly Express
        fig = px.line(crypto_data, x=crypto_data.index, y="Close",
                      title=f"{selected_crypto} Price ({time_range})")

        # Display line chart in Streamlit app
        st.plotly_chart(fig)
    elif choice == "Compare Crypto currency price":
        # Set up Streamlit app title
        st.title("Cryptocurrency price comparision chart")

        # Define list of available cryptocurrencies
        crypto_list = crypto_list_select

        # Display dropdown menu to select cryptocurrencies
        selected_cryptos = st.multiselect(
            "Select cryptocurrencies to compare", crypto_list)

        # Check if any cryptocurrencies are selected
        if not selected_cryptos:
            st.warning("Please select at least one cryptocurrency.")
        else:
            # Define time range for historical price data
            time_range = st.selectbox("Select time range", [
                "1d", "5d", "1mo", "6mo", "1y", "5y"])

            # Get historical price data using yfinance library
            crypto_data = yf.download(
                selected_cryptos, period=time_range)["Close"]

            # Create line chart of historical price data using Plotly Express
            fig = px.line(crypto_data, x=crypto_data.index, y=crypto_data.columns,
                          title="Cryptocurrency Price Comparison")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")

            # Display line chart in Streamlit app
            st.plotly_chart(fig)
    elif choice == "Run the project":
        st.title('Ethereum Price predicting using XGB model')
        # Create a button to generate the ETH dataset
        if st.button("Generate ETH Dataset"):
            eth_data = generate_eth_dataset()
            if eth_data is not None:
                st.write(eth_data)
                st.success("ETH dataset generated successfully!")
            else:
                st.error("Failed to generate ETH dataset.")
        if st.button("Processing the Dataset"):
            eth_data = generate_eth_dataset()
            if eth_data is not None:
                eth_data, X, y, X_train, X_test, y_train, y_test, dtrain, dtest = process_dataset(
                    eth_data)
                st.success("ETH dataset processing successfully!")
                st.success(
                    "Use the processed data as needed for further analysis")
            else:
                st.warning("Please generate the ETH dataset first.")
        if st.button("Build and train XGB model"):
            eth_data = generate_eth_dataset()
            if eth_data is not None:
                eth_data, X, y, X_train, X_test, y_train, y_test, dtrain, dtest = process_dataset(
                    eth_data)
            else:
                st.warning("Please generate the ETH dataset first.")
            xgb_model = train_xgboost_model(dtrain)
            st.success("XGBoost model trained successfully!")
        if st.button("Predict on Testing Dataset"):
            eth_data = generate_eth_dataset()

            if eth_data is not None:
                eth_data, X, y, X_train, X_test, y_train, y_test, dtrain, dtest = process_dataset(
                    eth_data)
            else:
                st.warning("Please generate the ETH dataset first.")

            xgb_model = train_xgboost_model(dtrain)

            prediction, y_test_transformed = predict_on_testing_dataset(
                xgb_model, dtest, y_test)
            df = pd.DataFrame(
                {'Prediction': prediction, 'Actual': y_test_transformed})
            fig = px.line(df, title='ETH PRICE PREDICTION',
                          labels={'index': 'Day', 'value': 'Price (USD)'})
            # Add hover information to display the values
            fig.update_traces(hovertemplate='Day: %{x}<br>Price: %{y}')
            # Display the line chart in Streamlit app
            st.plotly_chart(fig)
    elif choice == "Predict the lastest days":
        st.title('Predicting the latest days')
        # FOR TRAINING MODEL:
        eth_data = generate_eth_dataset()
        eth_data, X, y, X_train, X_test, y_train, y_test, dtrain, dtest = process_dataset(
            eth_data)
        xgb_model = train_xgboost_model(dtrain)
        # USE IT

        with st.form("prediction_form"):
            time_range = st.number_input(
                'Pick a latest time-range', 7, 1600)
            # Generated a second time ----------
            eth_data_last = generate_eth_dataset_last(time_range)
            X_last = eth_data_last.drop('Close', axis=1)
            y_last = eth_data_last['Close']
            # Create DMatrix for the previous data
            dxLast = xgb.DMatrix(X_last.values)

            form_submit_button = st.form_submit_button("Run Prediction")
            if form_submit_button:

                predictionLast, y_test_transformedLast = predict_lastest_days(
                    xgb_model, dxLast, y_last)
                df = pd.DataFrame(
                    {'Prediction': predictionLast, 'Actual': y_test_transformedLast})
                st.write(df)  # Display the prediction values
                fig = px.line(df, title=f"ETH PRICE PREDICTION of the last {time_range}",
                              labels={'index': 'Days', 'value': 'Price (USD)'})
                # Add hover information to display the values
                fig.update_traces(hovertemplate='Day: %{x}<br>Price: %{y}')

                # Display the line chart in Streamlit app

                st.plotly_chart(fig)


create_navbar()
