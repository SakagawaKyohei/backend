import psycopg2
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dotenv import load_dotenv
import os
from flask import Flask, jsonify

# Tải các biến môi trường từ file .env
load_dotenv()

# Lấy thông tin kết nối từ các biến môi trường
POSTGRES_HOST = os.getenv('POSTGRES_HOST1')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', 5432)  # Cổng mặc định PostgreSQL là 5432
POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')

app = Flask(__name__)

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

# Route mặc định
@app.route('/')
def home():
    return "Flask API is running"

# Route để dự đoán dữ liệu
@app.route('/forecast', methods=['GET'])
def forecast():
    # Lấy dữ liệu từ PostgreSQL
    data = get_data_from_db()

    # Dự đoán doanh thu trong tương lai
    forecast_result = forecast_data(data)

    # Trả về kết quả dự đoán dưới dạng JSON
    return jsonify(forecast_result)


# import os
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from flask import Flask, jsonify
# from dotenv import load_dotenv
# from sqlalchemy import create_engine

# # Tải các biến môi trường từ file .env
# load_dotenv()

# # Lấy thông tin kết nối từ các biến môi trường
# POSTGRES_HOST = os.getenv('POSTGRES_HOST1')
# POSTGRES_PORT = os.getenv('POSTGRES_PORT', 5432)  # Cổng mặc định PostgreSQL là 5432
# POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE')
# POSTGRES_USER = os.getenv('POSTGRES_USER')
# POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')

# app = Flask(__name__)

# # Hàm kết nối đến PostgreSQL qua SQLAlchemy
# def get_db_connection():
#     connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
#     engine = create_engine(connection_string)
#     return engine

# # Hàm lấy dữ liệu từ PostgreSQL
# def get_data_from_db():
#     engine = get_db_connection()
#     query = "SELECT year, month, revenue FROM doanhthu ORDER BY year, month"
#     df = pd.read_sql(query, engine)
#     return df

# # Hàm chuẩn bị dữ liệu cho Random Forest
# def prepare_data(data):
#     # Tạo cột `date` từ `year` và `month`
#     data['date'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str), format='%Y-%m')
#     data.set_index('date', inplace=True)

#     # Biến đổi dữ liệu thành dạng cần thiết cho mô hình học máy
#     data['prev_revenue'] = data['revenue'].shift(1)  # Thêm cột lag-1
#     data['prev_revenue_2'] = data['revenue'].shift(2)  # Thêm cột lag-2

#     # Loại bỏ các hàng NaN sau khi tạo cột lag
#     data.dropna(inplace=True)

#     # Các đặc trưng (features) và nhãn (target)
#     X = data[['year', 'month', 'prev_revenue', 'prev_revenue_2']]
#     y = data['revenue']

#     return X, y

# # Hàm dự đoán với mô hình Random Forest
# def forecast_data_rf(data, future_steps=6):
#     X, y = prepare_data(data)

#     # Chia dữ liệu thành tập huấn luyện và kiểm tra
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Chuẩn hóa dữ liệu
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Huấn luyện mô hình Random Forest
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     # Dự đoán các bước tương lai
#     future_revenues = []
#     last_row = X.iloc[-1].copy()  # Lấy hàng cuối cùng làm mẫu đầu vào

#     for _ in range(future_steps):
#         # Cập nhật tháng và năm
#         last_row['month'] = (last_row['month'] % 12) + 1
#         if last_row['month'] == 1:
#             last_row['year'] += 1

#         # Chuẩn bị dữ liệu để dự đoán
#         last_row_scaled = scaler.transform([last_row.values])
#         predicted_revenue = model.predict(last_row_scaled)[0]

#         # Thêm kết quả vào danh sách future_revenues
#         future_revenues.append(predicted_revenue)

#         # Cập nhật giá trị lag cho bước tiếp theo
#         last_row['prev_revenue_2'] = last_row['prev_revenue']
#         last_row['prev_revenue'] = predicted_revenue

#     # Trả về danh sách dự đoán doanh thu
#     return future_revenues

# # Route mặc định
# @app.route('/')
# def home():
#     return "Flask API is running"

# # Route để dự đoán dữ liệu
# @app.route('/forecast', methods=['GET'])
# def forecast():
#     # Lấy dữ liệu từ PostgreSQL
#     data = get_data_from_db()

#     # Dự đoán doanh thu trong tương lai
#     forecast_result = forecast_data_rf(data)

#     # Trả về kết quả dự đoán dưới dạng JSON
#     return jsonify(forecast_result)

# if __name__ == '__main__':
#     app.run(debug=True)

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
    product_names = request.json.get('product_names', [])

    if not product_names:
        return jsonify({"error": "No product names provided"}), 400

    all_forecasts = []

    # Loop through all product names to generate forecasts
    for product_name in product_names:
        # Get the training and forecast data for this product
        training_data = get_training_data_from_db(product_name)
        forecast_data = get_forecast_data_from_db()

        # Check if the data is empty
        if training_data.empty or forecast_data.empty:
            # Return a forecast of 0 if the data is empty
            all_forecasts.append({
                "product_name": product_name,
                "forecast_result": 0
            })
        else:
            # Proceed with the forecast calculation
            forecast_result = forecast_food_demand(training_data, forecast_data)
            all_forecasts.append({
                "product_name": product_name,
                "forecast_result": forecast_result
            })

    # Return all forecasts
    return jsonify({"forecasts": all_forecasts})

