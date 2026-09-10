# E-commerce Customer Behavior Analysis 🛒

A **comprehensive behavioural analytics dashboard** that analyses clickstream and transaction data to uncover session patterns, funnel drop-offs, cart abandonment behaviour, and purchase drivers.

Built with **Python + Streamlit** as a college final-year project.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Testing](#-testing)
- [Methodology](#-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [Screenshots](#-screenshots)
- [Advanced Enhancements](#-advanced-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Conversion Funnel** | Interactive funnel chart mapping View → Cart → Purchase with drop-off percentages |
| **Cohort Retention** | Monthly cohort heatmap tracking how purchase frequency evolves over time |
| **User Segmentation** | Automatic classification into Browsers, Cart Abandoners, and Purchasers |
| **K-Means Clustering** | Behaviour-based user clustering with 2-D scatter visualisation |
| **Time Analysis** | Hourly × daily purchase heatmap to identify peak conversion windows |
| **Brand Intelligence** | Top brands by views vs. purchases with conversion rate comparison |
| **KPI Dashboard** | Real-time metrics — total revenue, conversion rate, cart abandonment rate |
| **Interactive Filters** | Date range sliders and brand multi-select for dynamic exploration |
| **Sample Data Generator** | Works out-of-the-box with built-in synthetic data (no download needed) |

---

## 🏗️ System Architecture

```
📥 Clickstream Events (view, cart, purchase) + Transaction Data
     ↓
🧹 Session Construction & Bot Removal & Deduplication
     ↓
✅ Data Validation (schema, types, nulls)
     ↓
🔧 Funnel Stage Labelling
     ↓
📊 Conversion Rate Computation per Stage
     ↓
👥 Cohort Matrix Construction (by acquisition month)
     ↓
🔍 Behaviour Segmentation + K-Means Clustering
     ↓
🖥️ Funnel + Cohort + Segment + Time Dashboard (Streamlit)
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation & numerical computation |
| Plotly | Interactive charts (funnel, heatmaps, scatter) |
| Seaborn / Matplotlib | Statistical visualisations |
| Streamlit | Web-based dashboard framework |
| Scikit-learn | K-Means clustering & StandardScaler |
| SciPy | Chi-square statistical tests |
| python-dotenv | Environment variable management |

---

## 📁 Project Structure

```
ecommerce-behavior-analysis/
├── app.py                    # Streamlit dashboard (main entry point)
├── config.py                 # Centralised configuration & constants
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── test_validation.py        # Automated test suite (pytest)
├── README.md                 # This file
│
└── utils/                    # Core analysis modules
    ├── __init__.py            # Package exports
    ├── data_loader.py         # CSV loading & sample data generation
    ├── session_builder.py     # Session construction & bot removal
    ├── analyzer.py            # Funnel, cohort, segmentation, clustering
    ├── visualizer.py          # Plotly chart factory
    └── validator.py           # Data quality validation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ecommerce-behavior-analysis.git
cd ecommerce-behavior-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env
# Edit .env if you want to customise thresholds

# 5. Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📖 Usage

### Quick Start (Sample Data)

1. Launch the app: `streamlit run app.py`
2. Click **"🎲 Generate Sample Data"** in the sidebar
3. Explore all six dashboard tabs

### With Your Own Data

1. Prepare a CSV file with these **required columns**:
   | Column | Type | Description |
   |---|---|---|
   | `event_time` | datetime | Timestamp of the event |
   | `event_type` | string | One of: `view`, `cart`, `remove_from_cart`, `purchase` |
   | `product_id` | int/string | Product identifier |
   | `user_id` | int/string | User identifier |
   | `price` | float | Product price |

2. **Optional columns** (for richer analysis):
   | Column | Type | Description |
   |---|---|---|
   | `brand` | string | Product brand name |
   | `category_code` | string | Product category (e.g., `cosmetics.lip.lipstick`) |
   | `user_session` | string | Pre-assigned session ID |

3. Upload via the sidebar file uploader
4. Use date range sliders and brand filters to explore

### Recommended Datasets

| Dataset | Source | Size |
|---|---|---|
| eCommerce Events History in Cosmetics Shop | [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) | 2.7M events |
| Retail Rocket — E-commerce Dataset | [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | 2.7M events |
| Online Retail II Dataset | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) | 1M+ records |

---

## 🧪 Testing

Run the automated test suite:

```bash
# Run all tests with verbose output
python -m pytest test_validation.py -v

# Run a specific test class
python -m pytest test_validation.py::TestEndToEnd -v

# Run with coverage report
python -m pytest test_validation.py -v --cov=utils --cov-report=term-missing
```

### Test Coverage

| Test Class | What It Tests |
|---|---|
| `TestImports` | All library and module imports work |
| `TestConfig` | Configuration constants are valid |
| `TestSessionBuilder` | Bot removal, session construction, deduplication |
| `TestAnalyzer` | Funnel computation, cohort retention, segmentation |
| `TestEndToEnd` | Full pipeline from data generation to visualisation |

---

## 📐 Methodology

### 1. Funnel Analysis
- Each user is assigned the **maximum funnel stage** they reached: View → Cart → Purchase
- Conversion rates computed as the percentage of total viewers reaching each stage
- Drop-off rates show the loss between consecutive stages

### 2. Cohort Retention
- Users grouped by their **first activity month** (acquisition cohort)
- For each subsequent month, count how many cohort members are still active
- Displayed as a percentage heatmap (M0 = 100%, decay over time)

### 3. Behaviour Segmentation
- **Purchaser**: Completed at least one purchase
- **Cart Abandoner**: Added items to cart but never purchased
- **Browser**: Only viewed products (never carted or purchased)

### 4. K-Means Clustering
- Features: total_events, total_views, total_carts, total_purchases, avg_price, session_count
- StandardScaler normalisation → K-Means (k=4 by default)
- 2-D projection for visual exploration

### 5. Session Reconstruction
- Time-gap algorithm: >30 minutes between events = new session
- Bot detection: users exceeding 1,000 events/day are excluded

### 6. Statistical Testing
- Chi-square test available for comparing conversion rates across categorical groups

---

## 📊 Evaluation Metrics

| Metric | Formula |
|---|---|
| Session-to-Cart Rate | `users_who_carted / users_who_viewed × 100` |
| Cart-to-Purchase Rate | `users_who_purchased / users_who_carted × 100` |
| Overall Conversion Rate | `purchasers / total_viewers × 100` |
| Cart Abandonment Rate | `(carted − purchased) / carted × 100` |
| M1/M2/M3 Retention | `active_users_in_month_N / cohort_size × 100` |
| Avg Session Duration | Mean session length in seconds |
| Avg Products Viewed | Mean number of view events per session |

---

## 🖼️ Screenshots

After running the app, you'll see:

1. **📊 Overview Tab** — KPI cards, event distribution pie chart, daily event timeline
2. **🔻 Funnel Tab** — Interactive funnel chart with conversion percentages
3. **👥 Cohort Tab** — Monthly retention heatmap with cohort size comparison
4. **🧩 Segments Tab** — Donut chart of user segments, metrics comparison, K-Means scatter
5. **⏰ Time Tab** — Hour × day purchase heatmap, hourly event distribution
6. **🏷️ Brands Tab** — Top brands comparison chart, brand conversion metrics table

---

## 🚀 Advanced Enhancements

Future extensions that can be added:

- [ ] **Market Basket Analysis** — Co-purchase patterns using Apriori algorithm
- [ ] **A/B Test Calculator** — Statistical significance widget for checkout experiments
- [ ] **Cart Abandonment Prediction** — Gradient boosting classifier (XGBoost/LightGBM)
- [ ] **Google Analytics Integration** — Live data via GA4 API
- [ ] **Path Analysis** — Sankey diagram of most common navigation sequences
- [ ] **RFM Segmentation** — Recency, Frequency, Monetary value scoring

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

**Built  for learning and real-world analytics.**

