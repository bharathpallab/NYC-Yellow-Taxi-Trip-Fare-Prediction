# Tableau Dashboard & CSV Exports Guide

This guide provides a detailed reference of the exported CSV files located in your workspace's [csv_exports](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/) folder and details the step-by-step process for building the four interactive dashboards in Tableau.

---

## 📂 CSV Files Directory & Reference

All data sources for the dashboards are located in:
📂 **Path:** `c:\Users\shiva\OneDrive\Desktop\NYC Yellow Taxi Trip Records (January–June 2025),\csv_exports\`

### 1. Data Quality & Pipeline Metadata
*   **[dataset_overview.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/dataset_overview.csv):** General metadata for the dataset.
    *   *Columns:* `Metric`, `Value`
*   **[data_quality.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/data_quality.csv):** Ingestion pipeline cleaning steps and records retained.
    *   *Columns:* `Step`, `Records`, `Removed`
*   **[missing_values.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/missing_values.csv):** Count and percentage of missing values per column.
    *   *Columns:* `Column`, `Missing Count`, `Missing %`

### 2. Modeling & Performance Metrics
*   **[model_metrics.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/model_metrics.csv):** Key evaluation metrics across the four Spark models.
    *   *Columns:* `Model`, `Accuracy`, `Precision`, `Recall`, `F1 Score`, `ROC-AUC`, `PR-AUC`, `Specificity`, `Sensitivity`, `FPR`, `FNR`, `TP`, `TN`, `FP`, `FN`, `Training Time (s)`
*   **[feature_importance.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/feature_importance.csv):** Predictor importances extracted from the top-performing GBT Classifier.
    *   *Columns:* `feature`, `importance`
*   **[stability_results.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/stability_results.csv):** Model metrics under data reduction (90%, 80%) and random noise injection.
    *   *Columns:* `Scenario`, `Model`, `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`, `Time (s)`
*   **[class_distribution.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/class_distribution.csv):** Target variable distribution (`0 = Low Fare`, `1 = High Fare`).
    *   *Columns:* `Fare_Class`, `count`, `Percentage`, `Label`

### 3. Business Insights & Temporal Trends
*   **[business_insights.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/business_insights.csv):** Narrative takeaways mapped to specific categories.
    *   *Columns:* `Insight_No`, `Insight`, `Category`
*   **[hourly_fare_volume.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/hourly_fare_volume.csv):** Average fares and total volume for each hour of the day.
    *   *Columns:* `pickup_hour`, `avg_fare`, `total_revenue`, `trip_count`

### 4. Cost, Scalability & Spark Execution Performance
*   **[cost_analysis.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/cost_analysis.csv):** Fares, tips, tolls, and congestion surcharges averaged by month.
    *   *Columns:* `pickup_month`, `avg_fare`, `avg_tip`, `avg_tolls`, `avg_congestion`, `avg_total`, `total_revenue`
*   **[fare_distribution.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/fare_distribution.csv):** Distribution of fares grouped into $5 ranges.
    *   *Columns:* `fare_bin`, `count`, `fare_range`
*   **[execution_performance.csv](file:///c:/Users/shiva/OneDrive/Desktop/NYC%20Yellow%20Taxi%20Trip%20Records%20(January–June%202025),/csv_exports/execution_performance.csv):** Spark query execution comparison (Cached vs. Uncached).
    *   *Columns:* `Operation`, `Time (s)`, `Category`

---

## 🛠️ Step-by-Step Tableau Connection Guide

Since each CSV contains aggregated metrics suited to specific views, **do not attempt to join or relate these files in a single data source schema**. Instead, import them as separate, independent data sources.

1.  Open **Tableau Public** (or Tableau Desktop).
2.  Under **Connect**, select **Text File**.
3.  Navigate to `c:\Users\shiva\OneDrive\Desktop\NYC Yellow Taxi Trip Records (January–June 2025),\csv_exports\` and open the first file (e.g., `dataset_overview.csv`).
4.  To load the remaining files:
    *   Click the **Data** menu at the top -> **New Data Source**.
    *   Select **Text File** and load the next CSV.
    *   Repeat this for all 12 CSV files. You will see them listed in the top-left pane of your Worksheet under the Data tab.

---

## 📊 Dashboard Construction Steps

### 1️⃣ Dashboard 1: Data Quality & Pipeline Monitoring
This dashboard validates raw data ingestion, pipeline scrubbing efficiency, and missing value indicators.

#### Sheet 1: Ingestion Pipeline Funnel (Data Quality)
*   **Data Source:** `data_quality.csv`
*   **Rows:** `Step`
*   **Columns:** `Records`
*   **Mark Type:** Bar (horizontal).
*   **Formatting:** Sort `Step` manually to match the pipeline order:
    1. `Raw Ingestion`
    2. `Remove invalid fares (<=0 or >=200)`
    3. `Remove invalid distance (<=0 or >=100)`
    4. `Remove invalid passenger count`
    5. `Remove invalid duration (<30s or >2h)`
*   **Tooltip:** Add `Removed` to show the number of records filtered out in each step.
*   **Color:** Use a clean, professional gradient (e.g., deep steel blue to ice blue) to indicate volume reduction.

#### Sheet 2: Missingness Matrix
*   **Data Source:** `missing_values.csv`
*   **Rows:** `Column`
*   **Columns:** `Missing %` (Filtered to exclude columns with 0.0%)
*   **Mark Type:** Horizontal Bar chart.
*   **Sorting:** Sort `Column` descending by `Missing %`.
*   **Labels:** Enable label display for `Missing Count` and `Missing %` on the end of bars.

#### Sheet 3: Dataset Summary Cards (KPIs)
*   **Data Source:** `dataset_overview.csv`
*   **Cards:** Create a Text table card displaying `Metric` and `Value` as large KPI blocks (e.g., Total Records: **24.08M**, Total Columns: **20**, File Size: **0.38 GB**).

---

### 2️⃣ Dashboard 2: Model Performance & Feature Importance
Compares performance across models, visualizes predictive drivers, and evaluates stability.

#### Sheet 1: Model Comparison Chart
*   **Data Source:** `model_metrics.csv`
*   **Columns:** `Model`
*   **Rows:** `Measure Values` (Drag `Accuracy`, `F1 Score`, `Precision`, `Recall`, and `ROC-AUC` into the Measure Values card).
*   **Mark Type:** Side-by-side bars (grouped columns).
*   **Color:** Assign a unique color to each metric.
*   **Note:** Highlight the GBT Classifier as the best-performing model across all metrics (ROC-AUC: **0.9989**).

#### Sheet 2: Feature Importance Drivers
*   **Data Source:** `feature_importance.csv`
*   **Rows:** `feature`
*   **Columns:** `importance`
*   **Mark Type:** Horizontal Bar chart.
*   **Sorting:** Sort `feature` descending by `importance`.
*   **Color:** Highlight the top two columns (`trip_duration_sec` and `trip_distance`) in a distinct color to emphasize their combined importance (>95%).

#### Sheet 3: Model Stability Profile
*   **Data Source:** `stability_results.csv`
*   **Columns:** `Scenario` (Sort: Original -> 90% Data -> 80% Data -> Noise Injection)
*   **Rows:** `F1` or `ROC-AUC`
*   **Color:** `Model`
*   **Mark Type:** Line chart.
*   **Insight:** This chart shows that all models remain stable under data perturbation, with GBT holding a near-flat accuracy and ROC-AUC curve (highly resilient).

---

### 3️⃣ Dashboard 3: Business Insights & Temporal Trends
Displays pricing dynamics, trip volumes, and operational findings.

#### Sheet 1: Hourly Fare & Revenue Dual-Axis Chart
*   **Data Source:** `hourly_fare_volume.csv`
*   **Columns:** `pickup_hour` (Continuous/Dimension)
*   **Rows 1:** `avg_fare` (Line chart, left axis)
*   **Rows 2:** `trip_count` (Area or Bar chart, right axis)
*   **Configuration:** Right-click the second axis and select **Dual Axis**, then **Synchronize Axis** if applicable (or keep independent and style uniquely).
*   **Visual Highlights:** Annotate the peak hourly fare average during afternoon/evening rush hours (4:00 PM – 7:00 PM) and late-night airport transfers (midnight – 5:00 AM).

#### Sheet 2: Target Variable Class Balance
*   **Data Source:** `class_distribution.csv`
*   **Marks:** Donut chart or Pie chart using `Percentage` as angle and `Label` as color.
*   **Result:** Shows an almost 50/50 balance between **Low Fare** (50.78%) and **High Fare** (49.22%) classes.

#### Sheet 3: Executive Takeaways
*   **Data Source:** `business_insights.csv`
*   **Rows:** `Insight_No`, `Category`
*   **Detail:** `Insight`
*   **Format:** Present as a clean, structured table or bulletin list to summarize business conclusions directly on the dashboard.

---

### 4️⃣ Dashboard 4: Scalability & Cost Analysis
Analyzes cost breakdowns over time, passenger expenditures, and Spark execution times.

#### Sheet 1: Monthly Cost Component Breakdown
*   **Data Source:** `cost_analysis.csv`
*   **Columns:** `pickup_month` (formatted to display month names: January, February, etc.)
*   **Rows:** Drag `avg_fare`, `avg_tip`, `avg_tolls`, and `avg_congestion` into Rows.
*   **Mark Type:** Stacked Bar or Stacked Area chart.
*   **Color:** Color by Measure Names.
*   **Insight:** Shows the upward trend of fares and overall trip costs going into late spring.

#### Sheet 2: Fare Range Distribution
*   **Data Source:** `fare_distribution.csv`
*   **Columns:** `fare_range` (Sort manually or by `fare_bin` dimension)
*   **Rows:** `count`
*   **Mark Type:** Vertical Bar chart.
*   **Insight:** Shows that the majority of trips fall into the $5–$15 range, representing standard short-distance commuter trips.

#### Sheet 3: Spark Execution Performance (Cached vs. Uncached)
*   **Data Source:** `execution_performance.csv`
*   **Rows:** `Operation`
*   **Columns:** `Time (s)`
*   **Color:** `Category`
*   **Insight:** Shows how Spark caching reduces training and query time. For example, GBT training is cut in half from **5.6s** to **2.8s**.

---

## 🎬 Compiling the Tableau Story

To present a narrative flow to your stakeholders:
1.  Click **New Story** (the book icon next to the dashboard tab) in Tableau.
2.  Name the Story: **NYC Yellow Taxi Analytics & Modeling Story**.
3.  Create four Story Points, drag each dashboard into a separate point, and add captions:
    *   **Point 1: Ingestion & Data Quality:** *"How we cleaned 24 million records and managed missing data to feed our predictive pipeline."*
    *   **Point 2: Model Evaluation & Features:** *"Comparative performance of PySpark classifiers and key fare-class drivers."*
    *   **Point 3: Operational Demand & Insights:** *"Identifying Peak Hours, pricing distributions, and target class distributions."*
    *   **Point 4: Financial Dynamics & Performance:** *"Analyzing monthly cost breakdowns alongside Spark query optimizations."*
4.  Format the story navigation (e.g., using **Numbers** or **Captions**) for a smooth, interactive walkthrough.
