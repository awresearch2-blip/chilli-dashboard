# Chilli Intelligence Web -- production container image.
#
# Runs the exact same chilli_desktop business logic (data_loader, preprocessing,
# analytics, forecasting, insights) behind the Dash browser front end
# (chilli_web), served by waitress -- see run_web_production.py.
#
# Build:
#   docker build -t chilli-web .
# Run:
#   docker run -p 8060:8060 -e CHILLI_WORKBOOK=/app/data-source/workbook.xlsx chilli-web
# (omit -e CHILLI_WORKBOOK to use the workbook copied into the image below)

FROM python:3.13-slim

WORKDIR /app

# System deps needed to build/run scipy/statsmodels/xgboost wheels reliably.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY chilli_web/requirements.txt ./chilli_web/requirements.txt
RUN pip install --no-cache-dir -r chilli_web/requirements.txt

COPY chilli_desktop ./chilli_desktop
COPY chilli_web ./chilli_web
COPY run_web.py run_web_production.py ./
COPY "Chilli mastersheet for dashboard.xlsx" ./

# Writable dirs the app creates at runtime (logs, exports, Dash's diskcache).
RUN mkdir -p logs exports chilli_web/.diskcache

ENV HOST=0.0.0.0
ENV PORT=8060
EXPOSE 8060

CMD ["python", "run_web_production.py", "--host", "0.0.0.0", "--port", "8060"]
