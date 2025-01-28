import psycopg2
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dotenv import load_dotenv
import os
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load environment variables only in development
if os.path.exists('.env'):
    load_dotenv()

# Get database connection info from environment variables
POSTGRES_HOST = os.environ.get('POSTGRES_HOST1')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
POSTGRES_DATABASE = os.environ.get('POSTGRES_DATABASE')
POSTGRES_USER = os.environ.get('POSTGRES_USER')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')

app = Flask(__name__)

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Hàm kết nối đến PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DATABASE,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    return conn

# Hàm lấy dữ liệu từ PostgreSQL
def get_data_from_db():
    conn = get_db_connection()
    query = "SELECT year, month, revenue FROM doanhthu ORDER BY year, month"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Hàm dự đoán và trả về dự đoán dưới dạng JSON
def forecast_data(data, future_steps=6):
    # Chuyển đổi dữ liệu thành chuỗi thời gian (DateTime)
    data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m')
    data.set_index('date', inplace=True)

    # Đảm bảo dữ liệu có tần suất rõ ràng
    data = data['revenue'].asfreq('MS')  # 'MS' cho đầu mỗi tháng

    # Áp dụng mô hình Holt-Winters
    model_hw = ExponentialSmoothing(data, seasonal='multiplicative', seasonal_periods=12).fit()

    # Dự đoán trong tương lai
    forecast_hw = model_hw.forecast(steps=future_steps)

    # Chuyển kết quả dự đoán thành danh sách và trả về
    return forecast_hw.tolist()

# Add health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Modify your existing routes to remove the /api prefix since Render doesn't require it
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask API is running"})

@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        data = get_data_from_db()
        forecast_result = forecast_data(data)
        return jsonify(forecast_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Hàm lấy dữ liệu từ PostgreSQL cho việc huấn luyện
def get_training_data_from_db(product_name):
    engine = get_db_connection()
    query = f"""
SELECT 
    dt.id AS ma_doi_tac,
    ao_nuoi_item ->> 'thuysan' AS thuysan,
    ao_nuoi_item ->> 'soluong' AS soluong_thuysan,
    ao_nuoi_item ->> 'ngaytuoi' AS ngaytuoi_thuysan,
    product_item ->> 'id' AS ma_thucan,
    product_item ->> 'name' AS ten_thucan,
    product_item ->> 'soluong' AS soluong_thucan,
    TO_NUMBER(product_item ->> 'dongia', '999999999') AS dongia_thucan,
    TO_NUMBER(product_item ->> 'thanhtien', '999999999') AS thanhtien_thucan
FROM doitac dt
JOIN donxuathang dxh ON dt.id = dxh.ma_doi_tac
CROSS JOIN LATERAL unnest(dt.ao_nuoi) AS ao_nuoi_array
CROSS JOIN LATERAL jsonb_array_elements(ao_nuoi_array) AS ao_nuoi_item
CROSS JOIN LATERAL unnest(dxh.product) AS product_array
CROSS JOIN LATERAL jsonb_array_elements(product_array) AS product_item
WHERE TRIM(product_item ->> 'name') = '{product_name}'
ORDER BY dt.id, thuysan, ma_thucan;
    """
    df = pd.read_sql(query, engine)
    return df

# Hàm lấy dữ liệu từ PostgreSQL cho việc dự đoán
def get_forecast_data_from_db():
    engine = get_db_connection()
    query = """
    SELECT 
        ao_nuoi_item ->> 'thuysan' AS thuysan,
        ao_nuoi_item ->> 'ngaytuoi' AS ngaytuoi_thuysan,
        SUM((ao_nuoi_item ->> 'soluong')::NUMERIC) AS soluong_thuysan
    FROM doitac dt
    CROSS JOIN LATERAL unnest(dt.ao_nuoi) AS ao_nuoi_array
    CROSS JOIN LATERAL jsonb_array_elements(ao_nuoi_array) AS ao_nuoi_item
    GROUP BY thuysan, ngaytuoi_thuysan
    ORDER BY thuysan, ngaytuoi_thuysan;
    """
    df = pd.read_sql(query, engine)
    return df

# Hàm chuẩn bị dữ liệu cho Random Forest
def prepare_data(data):
    # Biến đổi dữ liệu thành dạng cần thiết cho mô hình học máy
    data['ngaytuoi_thuysan'] = data['ngaytuoi_thuysan'].astype(int)
    data['soluong_thuysan'] = data['soluong_thuysan'].astype(float)

    # Chuyển đổi cột "thuysan" thành dạng số sử dụng encoding
    data['thuysan'] = data['thuysan'].astype('category').cat.codes

    # Các đặc trưng (features) và nhãn (target)
    X = data[['thuysan', 'ngaytuoi_thuysan', 'soluong_thuysan']]
    y = data['soluong_thucan'] if 'soluong_thucan' in data.columns else None

    return X, y

# Hàm dự đoán với mô hình Random Forest
def forecast_food_demand(training_data, forecast_data):
    # Kiểm tra dữ liệu huấn luyện trước khi huấn luyện
    if training_data.empty:
        raise ValueError("Dữ liệu huấn luyện trống, không thể huấn luyện mô hình.")

    # Chuẩn bị dữ liệu huấn luyện
    X_train, y_train = prepare_data(training_data)

    # Kiểm tra xem X_train có đủ dữ liệu không để chia thành train/test
    if len(X_train) == 0:
        raise ValueError("Dữ liệu huấn luyện không đủ mẫu để chia thành train/test.")

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Huấn luyện mô hình Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán nhu cầu thức ăn cho dữ liệu dự báo (forecast_data)
    X_forecast, _ = prepare_data(forecast_data)
    X_forecast = scaler.transform(X_forecast)

    # Dự đoán
    y_forecast = model.predict(X_forecast)

    # Trả về tổng số dự đoán
    total_forecast = np.sum(y_forecast)
    return total_forecast

# Route để dự đoán nhu cầu thức ăn
@app.route('/demand', methods=['POST'])
def demand_forecast():
    try:
        product_names = request.json.get('product_names', [])
        if not product_names:
            return jsonify({"error": "No product names provided"}), 400

        all_forecasts = []
        for product_name in product_names:
            training_data = get_training_data_from_db(product_name)
            forecast_data = get_forecast_data_from_db()

            if training_data.empty or forecast_data.empty:
                all_forecasts.append({
                    "product_name": product_name,
                    "forecast_result": 0
                })
            else:
                forecast_result = forecast_food_demand(training_data, forecast_data)
                all_forecasts.append({
                    "product_name": product_name,
                    "forecast_result": float(forecast_result)
                })

        return jsonify({"forecasts": all_forecasts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


# import psycopg2
# import pandas as pd
# import pickle
# import numpy as np
# from flask import Flask, jsonify, request
# from sklearn.preprocessing import StandardScaler

# # Load environment variables
# import os
# from dotenv import load_dotenv
# if os.path.exists('.env'):
#     load_dotenv()

# # Get database connection info from environment variables
# POSTGRES_HOST = os.environ.get('POSTGRES_HOST1')
# POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
# POSTGRES_DATABASE = os.environ.get('POSTGRES_DATABASE')
# POSTGRES_USER = os.environ.get('POSTGRES_USER')
# POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD')

# app = Flask(__name__)

# # Add CORS support
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     return response

# # Hàm kết nối đến PostgreSQL
# def get_db_connection():
#     conn = psycopg2.connect(
#         host=POSTGRES_HOST,
#         port=POSTGRES_PORT,
#         database=POSTGRES_DATABASE,
#         user=POSTGRES_USER,
#         password=POSTGRES_PASSWORD
#     )
#     return conn

# # Hàm lấy dữ liệu từ PostgreSQL
# def get_data_from_db():
#     conn = get_db_connection()
#     query = "SELECT year, month, revenue FROM doanhthu ORDER BY year, month"
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# # Hàm dự đoán doanh thu từ mô hình đã lưu
# def forecast_revenue(data, future_steps=6):
#     # Tải mô hình đã lưu
#     with open('static/revenueforecast.pkl', 'rb') as file:
#         model_hw = pickle.load(file)

#     # Chuyển dữ liệu thành chuỗi thời gian
#     data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m')
#     data.set_index('date', inplace=True)

#     # Đảm bảo dữ liệu có tần suất rõ ràng
#     data = data['revenue'].asfreq('MS')

#     # Dự đoán doanh thu trong tương lai
#     forecast_hw = model_hw.forecast(steps=future_steps)
    
#     return forecast_hw.tolist()

# # Hàm dự đoán nhu cầu thức ăn từ mô hình đã lưu
# def forecast_food_demand(training_data, forecast_data):
#     # Tải mô hình đã lưu
#     with open('static/demandforecast.pkl', 'rb') as file:
#         model = pickle.load(file)

#     # Chuẩn bị dữ liệu huấn luyện
#     X_train, y_train = prepare_data(training_data)
#     if len(X_train) == 0:
#         raise ValueError("Dữ liệu huấn luyện không đủ mẫu để chia thành train/test.")

#     # Chuẩn hóa dữ liệu
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)

#     # Dự đoán nhu cầu thức ăn cho dữ liệu dự báo (forecast_data)
#     X_forecast, _ = prepare_data(forecast_data)
#     X_forecast = scaler.transform(X_forecast)

#     # Dự đoán với mô hình đã lưu
#     y_forecast = model.predict(X_forecast)

#     # Trả về tổng số dự đoán
#     total_forecast = np.sum(y_forecast)
#     return total_forecast

# # Route để dự đoán doanh thu
# @app.route('/forecast', methods=['GET'])
# def forecast():
#     try:
#         data = get_data_from_db()
#         forecast_result = forecast_revenue(data)
#         return jsonify(forecast_result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Route để dự đoán nhu cầu thức ăn
# @app.route('/demand', methods=['POST'])
# def demand_forecast():
#     try:
#         product_names = request.json.get('product_names', [])
#         if not product_names:
#             return jsonify({"error": "No product names provided"}), 400

#         all_forecasts = []
#         for product_name in product_names:
#             training_data = get_training_data_from_db(product_name)
#             forecast_data = get_forecast_data_from_db()

#             if training_data.empty or forecast_data.empty:
#                 all_forecasts.append({
#                     "product_name": product_name,
#                     "forecast_result": 0
#                 })
#             else:
#                 forecast_result = forecast_food_demand(training_data, forecast_data)
#                 all_forecasts.append({
#                     "product_name": product_name,
#                     "forecast_result": float(forecast_result)
#                 })

#         return jsonify({"forecasts": all_forecasts})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)

