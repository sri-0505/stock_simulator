import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import json
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

st.set_page_config(page_title="Stock Simulator", layout="wide")

# Load or initialize portfolio
if os.path.exists("portfolio.json"):
    with open("portfolio.json", "r") as f:
        portfolio = json.load(f)
else:
    portfolio = {}

st.title("📈 Stock Market Simulator + Visual Tracker")

# Sidebar Navigation
page = st.sidebar.radio("Go to", ["📊 Dashboard", "📈 Compare","👥 Buy", "👤 Sell", "📁 Portfolio", "📐 Fibonacci"])

# Common Inputs
# 🔽 Custom + Dropdown Ticker Input
st.sidebar.subheader("📌 Select or Enter Stock Ticker")

# Dropdown list of popular stocks
popular_tickers = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",  # Indian stocks
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"                             # US stocks
]

# Checkbox to choose manual entry
use_custom = st.sidebar.checkbox("✏️ Type custom ticker instead")

# Conditional input
if use_custom:
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS)", "TCS.NS")
else:
    ticker = st.sidebar.selectbox("Select Stock Ticker", popular_tickers, index=0)

start_date = st.sidebar.date_input("From", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("To", pd.to_datetime("today"))

# Page: Dashboard
if page == "📊 Dashboard":
    if st.sidebar.button("Load Data"):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                st.error("❌ No data found. Please check the ticker symbol or date range.")
            else:
                st.success(f"✅ Stock Data for {ticker}")

                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()

                st.subheader("📈 Close Price with Moving Averages")
                st.markdown("📘 *Legend*:\- 🔵 Blue = Close Price\- 🟠 Orange = 20-day MA\- 🟢 Green = 50-day MA")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20 (20-day)', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50 (50-day)', line=dict(color='green')))
                fig.update_layout(
                    title="📈 Close Price + Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    autosize=True,
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h"),  # better for mobile
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🔓 Candlestick Chart")
                st.info("🟢 Green = Price went up (Close > Open), 🔴 Red = Price went down (Close < Open)")

                fig2 = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    name='Candlestick'
                )])
                fig2.update_layout(
                    title="🔓 Candlestick Chart", 
                    xaxis_title="Date", 
                    yaxis_title="Price",
                    autosize=True,
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h"),  # better for mobile
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("📦 Trading Volume")
                st.info("🟢 Green = Close > Open (Bullish volume), 🔴 Red = Close < Open (Bearish volume)")

                colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red' for i in range(len(data))]
                fig3 = go.Figure([go.Bar(x=data.index, y=data['Volume'], marker_color=colors)])
                fig3.update_layout( 
                    title="📦 Trading Volume", 
                    xaxis_title="Date", 
                    yaxis_title="Volume",
                    autosize=True,
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h"),  # better for mobile
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

#Compare block
elif page == "📈 Compare":
    st.header("📈 Compare Multiple Stock Prices")

    # Select up to 3 stock tickers
    tickers = st.multiselect(
        "Select up to 3 stocks to compare:",
        options=[
            "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",  # Indian stocks
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"                             # US stocks
        ],
        default=["TCS.NS"],
        max_selections=3
    )

    compare_start = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    compare_end = st.date_input("End Date", datetime.now())

    # Define custom colors
    color_map = ['blue', 'lightblue', 'red', 'green', 'orange', 'purple']

    if tickers:
        st.subheader("📉 Closing Price Comparison")
        price_fig = go.Figure()
        norm_fig = go.Figure()
        performance_summary = {}

        for i, symbol in enumerate(tickers):
            try:
                df = yf.download(symbol, start=compare_start, end=compare_end)
                if not df.empty:
                    # --- Closing Price Chart ---
                    price_fig.add_trace(go.Scatter(
                        x=pd.to_datetime(df.index),
                        y=df["Close"].round(2),
                        mode="lines",
                        name=symbol,
                        line=dict(color=color_map[i % len(color_map)], width=3)
                    ))

                    # --- Normalized Chart ---
                    norm = df["Close"] / df["Close"].iloc[0] * 100
                    norm_fig.add_trace(go.Scatter(
                        x=pd.to_datetime(df.index),
                        y=norm.round(2),
                        mode="lines",
                        name=symbol,
                        line=dict(color=color_map[i % len(color_map)], width=3)
                    ))

                    # --- Performance Calculation ---
                    start_price = df["Close"].iloc[0]
                    end_price = df["Close"].iloc[-1]
                    percent_return = ((end_price - start_price) / start_price) * 100
                    performance_summary[symbol] = percent_return

                else:
                    st.warning(f"⚠️ No data available for {symbol}. Skipping.")

            except Exception as e:
                st.error(f"❌ Error loading {symbol}: {e}")

        # Show Price Comparison Chart
        price_fig.update_layout(
            title="Stock Prices Over Time", 
            xaxis_title="Date",
            yaxis_title="Price",
            autosize=True,
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h"),  # better for mobile
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(price_fig, use_container_width=True)

        # Show Normalized Comparison Chart
        st.subheader("📈 Normalized Performance (Base = 100)")
        norm_fig.update_layout(
            title="Normalized Stock Performance", 
            xaxis_title="Date", 
            yaxis_title="Performance Index",
            autosize=True,
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h"),  # better for mobile
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(norm_fig, use_container_width=True)

        # --- Final Recommendation ---
        if performance_summary:
            for stock, summary in performance_summary.items():
                print(f"{stock}: {summary}")

            best_stock = max(performance_summary, key=lambda x: performance_summary[x].values[0])
            best_return = performance_summary[best_stock].values[0]
            st.success(
                f"💡 **Recommendation**: Based on the selected period, **{best_stock}** has the best return of **{best_return:.2f}%**."
            )
        else:
            st.warning("⚠️ Could not compute performance due to missing data.")

    else:
        st.info("👆 Please select at least one stock to compare.")

# Page: Buy
elif page == "👥 Buy":
    st.subheader("👥 Buy Stocks")

    st.write(f"Buying for: {ticker}")
    quantity = st.number_input("Enter Quantity", min_value=1, value=1)
    predict_days = st.slider("Predict Future Price for (days)", 1, 30, 7)

    if st.button("Predict Before Buying"):
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data found. Please check the ticker or date range.")
            st.stop()

        df = data.copy()
        df['Date'] = pd.to_datetime(df.index)
        df = df[['Date', 'Close']].dropna()
        y_actual = df['Close'].values
        current_price = float(y_actual[-1])
        currency = "₹" if ticker.endswith(".NS") else "$"
        future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=predict_days)

        predictions = {}
        roi_results = {}

        # ----- Linear Regression -----
        model_lr = LinearRegression()
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['Close'].values
        model_lr.fit(X, y)
        future_X = np.array(range(len(df), len(df) + predict_days)).reshape(-1, 1)
        pred_lr = model_lr.predict(future_X).flatten()
        predictions['Linear Regression'] = pred_lr
        roi_results['Linear Regression'] = ((pred_lr[-1] - current_price) / current_price) * 100

        # ----- Random Forest -----
        model_rf = RandomForestRegressor(n_estimators=100)
        model_rf.fit(X, y)
        pred_rf = model_rf.predict(future_X).flatten()
        predictions['Random Forest'] = pred_rf
        roi_results['Random Forest'] = ((pred_rf[-1] - current_price) / current_price) * 100

        # ----- LSTM -----
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        X_lstm, y_lstm = [], []
        for i in range(60, len(scaled_data)):
            X_lstm.append(scaled_data[i - 60:i, 0])
            y_lstm.append(scaled_data[i, 0])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

        pred_lstm = []
        if len(X_lstm) > 0:
            model_lstm = Sequential()
            model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
            model_lstm.add(LSTM(units=50))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0)

            test_input = scaled_data[-60:]
            for _ in range(predict_days):
                pred = model_lstm.predict(test_input.reshape(1, 60, 1), verbose=0)
                pred_lstm.append(pred[0][0])
                test_input = np.append(test_input[1:], pred).reshape(60, 1)

            pred_lstm = scaler.inverse_transform(np.array(pred_lstm).reshape(-1, 1)).flatten()
            predictions['LSTM'] = pred_lstm
            roi_results['LSTM'] = ((pred_lstm[-1] - current_price) / current_price) * 100
        else:
            pred_lstm = np.full(predict_days, current_price)
            predictions['LSTM'] = pred_lstm
            roi_results['LSTM'] = 0.0

        st.metric("Current Price", f"{currency}{current_price:.2f}")

        model_colors = {
            "Linear Regression": "orange",
            "Random Forest": "green",
            "LSTM": "purple"
        }

        for model_name, pred in predictions.items():
            st.subheader(f"📉 {model_name} Forecast")
            fig_model = go.Figure()
            fig_model.add_trace(go.Scatter(x=df['Date'], y=y_actual, mode='lines', name='Historical', line=dict(color='gray')))
            fig_model.add_trace(go.Scatter(x=future_dates, y=pred, mode='lines+markers', name=f'{model_name} Forecast', line=dict(color=model_colors[model_name], width=2)))
            fig_model.update_layout(title=f"{model_name} - Historical + Forecast", xaxis_title="Date", yaxis_title="Price", height=350, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig_model, use_container_width=True)

        st.markdown("## 📊 ROI Projection per Model")
        for model_name, pred in predictions.items():
            future_roi = [(p - current_price) / current_price * 100 for p in pred]
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(x=future_dates, y=future_roi, mode='lines+markers', name='Projected ROI', line=dict(color=model_colors[model_name], width=2)))
            fig_roi.update_layout(title=f"{model_name} ROI (%)", xaxis_title="Date", yaxis_title="ROI (%)", height=300, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig_roi, use_container_width=True)

        st.markdown("## 📋 Model Comparison Summary")
        summary_data = []
        for model_name, pred in predictions.items():
            future_price = float(pred[-1])
            roi = roi_results[model_name]
            summary_data.append({"Model": model_name, "Predicted Price": f"{currency}{future_price:.2f}", "Projected ROI (%)": round(roi, 2)})
        df_summary = pd.DataFrame(summary_data)
        def highlight_roi(val): return 'color: green' if val > 0 else 'color: red'
        st.dataframe(df_summary.style.applymap(highlight_roi, subset=["Projected ROI (%)"]))

        best_model = max(roi_results, key=roi_results.get)
        best_roi = roi_results[best_model]
        best_price = predictions[best_model][-1]

        st.markdown("## ✅ Final Suggestion")
        if best_roi > 5:
            st.success(f"📈 **Recommendation**: Based on prediction, **{best_model}** gives best ROI: **{best_roi:.2f}%**\n\nPredicted price: **{currency}{best_price:.2f}**")
        elif best_roi > 0:
            st.info(f"⚠️ ROI is positive but moderate. Best model: **{best_model}** with ROI **{best_roi:.2f}%**.")
        else:
            st.warning(f"📉 All models predict decline. Best option is **{best_model}**, but ROI is negative: **{best_roi:.2f}%**. Consider waiting.")

        st.session_state.current_price = current_price
        st.session_state.currency = currency

    if st.button("Confirm Purchase"):
        if "current_price" not in st.session_state:
            st.error("⚠ Please predict the stock price first before purchasing.")
        else:
            current_price = st.session_state.current_price
            currency = st.session_state.currency
            try:
                with open("portfolio.json", "r") as f:
                    portfolio = json.load(f)
            except FileNotFoundError:
                portfolio = {}

            if ticker in portfolio:
                portfolio[ticker]['shares'] += quantity
            else:
                portfolio[ticker] = {
                    "shares": quantity,
                    "bought_price": current_price,
                    "invested": current_price * quantity
                }

            with open("portfolio.json", "w") as f:
                json.dump(portfolio, f)

            st.success(f"✅ Bought {quantity} shares of {ticker} at {currency}{current_price:.2f}")

# Page: Sell
elif page == "👤 Sell":
    st.header("👤 Sell Stocks")

    if ticker not in portfolio:
        try:
            with open("portfolio.json","r") as f:
                portfolio = json.load(f)
        except FileNotFoundError:
            portfolio = {}
        st.warning("❌ You don't own this stock.")
    else:
        available_shares = portfolio[ticker]["shares"]
        st.write(f"You own {available_shares} shares of {ticker}")
        sell_quantity = st.number_input("Enter quantity to sell:", min_value=1, max_value=available_shares, step=1)

        if st.button("Sell Now"):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")

                if data.empty:
                    st.warning("⚠️ No price data available.")
                else:
                    price = round(data["Close"].iloc[-1], 2)
                    revenue = price * sell_quantity

                    avg_price = portfolio[ticker]["invested"] / (portfolio[ticker]["shares"])
                    portfolio[ticker]["shares"] -= sell_quantity
                    portfolio[ticker]["invested"] -= avg_price * sell_quantity

                    if portfolio[ticker]["shares"] == 0:
                        del portfolio[ticker]

                    with open("portfolio.json", "w") as f:
                        json.dump(portfolio, f)

                    st.success(f"✅ Sold {sell_quantity} shares of {ticker} at ₹{price} (₹{revenue:.2f})")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Page: Portfolio
elif page == "📁 Portfolio":
    st.header("📁 My Portfolio")

    if not portfolio:
        st.warning("No investments yet.")
    else:
        rows = []
        for symbol, info in portfolio.items():
            try:
                stock = yf.Ticker(symbol)
                price_data = stock.history(start=datetime.today() - timedelta(days=5), end=datetime.today())
                price_data = price_data[price_data["Close"].notna()]
                if price_data.empty:
                    st.warning(f"No data for {symbol}")
                    continue
                current_price = price_data["Close"].iloc[-1]
                total_value = info["shares"] * current_price
                invested = info["invested"]
                roi = (total_value - invested) / invested * 100
                rows.append({
                    "Ticker": symbol,
                    "Shares": info["shares"],
                    "Invested (₹)": round(invested, 2),
                    "Current Value (₹)": round(total_value, 2),
                    "ROI (%)": round(roi, 2)
                })
            except Exception as e:
                st.warning(f"Could not fetch data for {symbol}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(by="ROI (%)", ascending=False)

            def color_roi(val):
                return 'color: green' if val > 0 else 'color: red'

            st.dataframe(df.style.applymap(color_roi, subset=["ROI (%)"]))

            total_invested = sum(r["Invested (₹)"] for r in rows)
            total_value = sum(r["Current Value (₹)"] for r in rows)
            total_roi = (total_value - total_invested) / total_invested * 100

            st.subheader("📊 Portfolio Summary")
            st.metric("Total Investment", f"₹{total_invested:.2f}")
            st.metric("Current Value", f"₹{total_value:.2f}")
            st.metric("Overall ROI", f"{total_roi:.2f}%")

            st.subheader("📉 ROI Chart")
            fig3 = go.Figure([go.Bar(
                x=df["Ticker"], 
                y=df["ROI (%)"],
                marker_color=['green' if val > 0 else 'red' for val in df["ROI (%)"]]
            )])
            fig3.update_layout(
                title="return on investment by ticker",
                yaxis_title="roi(%)",
                autosize=True,
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h"),  # better for mobile
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig3,use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Portfolio", csv, "portfolio.csv", "text/csv") 

#Fibonacci
elif page == "📐 Fibonacci":
    st.header("📐 Fibonacci Retracement Calculator")

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            st.warning("❌ No data found for the selected range.")
        else:
            high_price = data['High'].max()
            low_price = data['Low'].min()

            diff = high_price - low_price
            levels = {
                "0.0%": high_price,
                "23.6%": high_price - 0.236 * diff,
                "38.2%": high_price - 0.382 * diff,
                "50.0%": high_price - 0.5 * diff,
                "61.8%": high_price - 0.618 * diff,
                "78.6%": high_price - 0.786 * diff,
                "100.0%": low_price
            }

            st.subheader(f"High: ₹{high_price:.2f} | Low: ₹{low_price:.2f}")
            df_levels = pd.DataFrame(list(levels.items()), columns=["Level", "Price"])
            st.dataframe(df_levels.style.format({"Price": "{:.2f}"}))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))

            for level, price in levels.items():
                fig.add_hline(
                    y=price, 
                    line_dash="dot", 
                    annotation_text=level,
                    annotation_position="right", 
                    line=dict(color="orange")
                )

            fig.update_layout(
                title="Fibonacci Retracement Levels", 
                yaxis_title="Price (₹)",
                autosize=True,
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h"),  # better for mobile
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
# trigger rebuild
