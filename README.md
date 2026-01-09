# Rossmann Store Sales Forecasting

## Overview

This project tackles the **Rossmann Sales Forecasting Challenge**, which involves predicting 6 weeks of daily sales for 1,115 drug stores located across Germany. Rossmann operates over 3,000 drug stores in 7 European countries, and accurate sales forecasts are critical for creating effective staff schedules that increase productivity and motivation.

## Business Problem

Store sales are influenced by many complex factors including:
- Promotions (Promo and Promo2)
- Competition proximity and history
- School and state holidays
- Seasonality (weekly, monthly, yearly patterns)
- Store locality and characteristics

With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied. This project aims to create a robust prediction model using machine learning to support data-driven decision making.

## Evaluation Metric

Submissions are evaluated on **RMSPE** (Root Mean Square Percentage Error):

```
RMSPE = sqrt(1/n * Σ((y_i - ŷ_i)/y_i)²)
```

Where:
- `y_i` = actual sales of a single store on a single day
- `ŷ_i` = predicted sales
- Days with 0 sales are ignored in scoring

## Dataset Description

### Training Data
- **Records**: 1,017,209 rows
- **Features**: 9 columns
- **Time Period**: 2013-2015 (2+ years of historical data)

### Test Data
- **Records**: 41,088 rows (before filtering)
- **Final Predictions**: 35,093 rows (open stores only)
- **Stores**: 856 stores with open days in test period
- **Prediction Period**: 6 weeks (September-October 2015)

### Store Metadata
- **Stores**: 1,115 unique stores
- **Store Types**: a, b, c, d
- **Assortment Types**: basic (a), extended (b), extra (c)

### Key Features
- **Store**: Unique store identifier
- **DayOfWeek**: Day of week (1=Monday, 7=Sunday)
- **Date**: Transaction date
- **Sales**: Sales revenue (target variable)
- **Customers**: Number of customers
- **Open**: Store open status (0=closed, 1=open)
- **Promo**: Standard promotion indicator
- **StateHoliday**: State holiday indicator (0, a, b, c)
- **SchoolHoliday**: School holiday indicator
- **StoreType**: Store format (a, b, c, d)
- **Assortment**: Product range (a, b, c)
- **CompetitionDistance**: Distance to nearest competitor (meters)
- **CompetitionOpenSinceMonth/Year**: Competition opening date
- **Promo2**: Extended promotion participation
- **Promo2SinceWeek/Year**: Extended promotion start date
- **PromoInterval**: Months when Promo2 is active

## Project Structure

```
Rossmann_Forecasting/
│
├── Preprocessing&Testing.ipynb          # EDA, preprocessing, model training
├── TestPrediction_XGBoost.ipynb        # Final predictions on test set
├── rossmann_xgb_platinum.json          # Trained XGBoost model (351MB)
├── store_sales_forecast_6weeks.csv     # Final predictions output
└── README.md                            # This file
```

## Methodology

### 1. Exploratory Data Analysis

**Data Characteristics:**
- Sales Distribution: Right-skewed (median ~5,744€)
- Mean Daily Sales: ~5,773.82€
- Mean Customers: ~633.15 per store-day
- Store Open Rate: 83%
- Promotional Coverage: 38.2%
- School Holidays: 17.9% of records

**Key Insights:**
- Strong day-of-week seasonality (weekends differ from weekdays)
- Promotions show consistent sales uplift
- Competition distance materially affects sales
- Store type and assortment level drive significant variation
- State and school holidays create distinct patterns

### 2. Data Preprocessing

#### Missing Value Imputation

**Competition Features (37% missing):**
- **Cluster 1** - Stores without competitors:
  - `CompetitionDistance` → 2× max observed distance (151,720m)
  - `CompetitionOpenSinceYear` → 2015
  - `CompetitionOpenSinceMonth` → 12

- **Cluster 2** - Stores with competitors but unknown opening date:
  - `CompetitionOpenSinceYear` → Median (2010)
  - `CompetitionOpenSinceMonth` → Median (7)

**Promo2 Features (42% missing):**
- Stores not participating in Promo2 (where `Promo2=0`):
  - `Promo2SinceWeek` → 0
  - `Promo2SinceYear` → 0
  - `PromoInterval` → '0'

**Open Status (0.03% missing):**
- Imputed missing values in test set

#### Data Quality

**StateHoliday Alignment:**
- Converted numeric `0` to string `'0'` for consistency with categorical values ('a', 'b', 'c')

**Store Filtering:**
- Removed closed stores (`Open=0`) from final predictions
- Final test set: 35,093 records (85% of original)

### 3. Feature Engineering

#### Temporal Features
```python
- Year: Long-term trend capture
- Month: Seasonal cycles (1-12)
- WeekOfYear: Finer seasonal patterns (1-52)
- DayOfMonth: Payday effect analysis (1-31)
```

#### Competition Intelligence
```python
- LifeTimeCompetition: Days since competitor opened
  - Benchmark: 2015-12-01
  - Calculation: (Benchmark - CompetitionOpenDate).days
  - Cap at 0 for future openings
  - Mean: 2,285 days (6.3 years)
```

#### Promotion Intelligence
```python
- LifeTimePromo2: Days since Promo2 enrollment
  - Calculated from ISO week format (Year + Week + Monday)
  - 0 for non-participating stores
  - Mean: 770 days (2.1 years) for participating stores

- isPromo2Month: Current month in PromoInterval (binary)
  - Maps Date.month to PromoInterval string
  - 12.7% of records are Promo2 months
```

#### Distance Transformation
```python
- LogCompetitionDistance: Log-transformed distance
  - Formula: log(1 + CompetitionDistance)
  - Reduces skewness: 7.819 → -0.390
  - Better model linearity
```

#### Categorical Encoding

**One-Hot Encoding:**
- `StoreType`: → StoreType_a, StoreType_b, StoreType_c, StoreType_d
- `Assortment`: → Assortment_a, Assortment_b, Assortment_c
- `StateHoliday`: → StateHoliday_0, StateHoliday_a, StateHoliday_b, StateHoliday_c

**Binary Features (already numeric):**
- Promo, SchoolHoliday, isPromo2Month

**Features Removed:**
- `Date`: Extracted to temporal components
- `Open`: Constant after filtering (all 1)
- `Promo2`: Redundant (captured in LifeTimePromo2)
- `CompetitionDistance`: Replaced by log-transformed version
- `CompetitionOpenSince*`, `Promo2Since*`: Replaced by LifeTime features
- `PromoInterval`: Replaced by isPromo2Month

### 4. Model Development

#### Algorithm: XGBoost (Extreme Gradient Boosting)

**Model Selection Rationale:**
- Handles non-linear relationships effectively
- Robust to outliers and missing values
- Captures complex feature interactions
- Fast training and prediction
- Built-in feature importance
- Excellent performance on tabular data

**Model Variants Tested:**
- Multiple XGBoost configurations explored
- Best performing: **PLATINUM Model**

**Final Model:**
- File: `rossmann_xgb_platinum.json` (351MB)
- Features: 23 engineered features
- Format: XGBoost native JSON

#### Feature Alignment

**Test Set Handling:**
- Missing categorical columns added with value 0
  - Example: `StateHoliday_b`, `StateHoliday_c` (not present in test period)
- Final feature count: 23 (matches training)
- Feature order preserved for model compatibility

### 5. Predictions

#### Output Format

**Raw Predictions:**
- Log-transformed predictions converted to Euro values
- Formula: `Sales = exp(log_prediction) - 1`

**Final Output File:** `store_sales_forecast_6weeks.csv`
- **Format**: Store ID, Total Sales (6-week forecast)
- **Records**: 856 stores
- **Total Forecasted Sales**: €243,624,340
- **Sorted by**: Sales (descending)

**Sample Output:**
```
Store    Sales
262      967,447.19
1114     866,129.75
562      830,730.75
251      749,968.75
733      740,507.06
```

## Model Performance

### PLATINUM Model Metrics

**RMSPE (Root Mean Square Percentage Error): 12.49%**

This represents strong performance for demand forecasting, indicating:
- Predictions deviate by ~12.5% on average
- Competitive accuracy for retail forecasting
- Suitable for operational planning and staffing decisions

### Validation Strategy

- **Time-series cross-validation**: Respects temporal ordering
- **Train-test split**: Chronological separation
- **Separate test set**: Final evaluation on unseen data

## Key Findings & Business Insights

### Weekly Patterns
- **Peak Days**: Mid-week shows highest average sales
- **Weekend Effect**: Distinct patterns on Saturdays and Sundays
- **Monday Minimum**: Typically lowest sales day

### Promotional Impact
- **Promo Effect**: Significant sales uplift during active promotions
- **Promo2 Months**: Extended promotions create sustained increases
- **Peak Correlation**: All top 3 sales peaks coincide with promotional days

### Competition Dynamics
- **Distance Effect**: Closer competitors correlate with varying sales patterns
- **Agglomeration Benefit**: Some stores benefit from nearby competition (shopping districts)
- **Lifetime Impact**: Longer competitor presence shows adaptation patterns

### Store Segmentation
- **Store Type**: Types show distinct sales distributions
- **Assortment**: Product range drives customer traffic differences
- **Geographic**: Location-based patterns evident in forecasts

## Visualizations

The notebooks include comprehensive visualizations:

1. **Forecast Trend Analysis**
   - Time series of predicted sales
   - Peak day annotations with promo correlation
   - Minimum day identification

2. **Weekly Distribution**
   - Boxplots by day of week
   - Mean ± standard deviation trends
   - Statistical summaries

3. **Competition Analysis**
   - Sales by competition distance bins
   - Agglomeration effect verification

4. **Store Performance**
   - Sales distribution by store type
   - Promotional impact comparisons

## Requirements

### Python Libraries
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
scikit-learn>=0.24.0
scipy>=1.7.0
```

### Data Files (Not Included)
```
rossmann-store-sales/
├── train.csv          # Historical sales data
├── test.csv           # Test period data
└── store.csv          # Store metadata
```

## Usage

### 1. Training and Preprocessing
```python
# Run the preprocessing and training notebook
jupyter notebook "Preprocessing&Testing.ipynb"
```

This notebook:
- Loads and explores the datasets
- Performs feature engineering
- Trains the XGBoost model
- Validates performance
- Saves the trained model

### 2. Generate Predictions
```python
# Run the prediction notebook
jupyter notebook "TestPrediction_XGBoost.ipynb"
```

This notebook:
- Loads the test data
- Applies the same preprocessing pipeline
- Loads the trained PLATINUM model
- Generates predictions
- Creates visualizations
- Exports results to CSV

### 3. Output
The final predictions are saved to:
```
store_sales_forecast_6weeks.csv
```

## Model Interpretation

### Feature Importance (Top Contributors)
Based on XGBoost's built-in feature importance:
1. Temporal features (DayOfWeek, Month, WeekOfYear)
2. Store characteristics (StoreType, Assortment)
3. Promotional indicators (Promo, isPromo2Month)
4. Competition metrics (LogCompetitionDistance, LifeTimeCompetition)
5. Holiday indicators (StateHoliday, SchoolHoliday)

### Prediction Confidence
- Higher confidence for typical weekdays with historical patterns
- Lower confidence for holiday periods and special events
- Promotional periods well-captured due to strong training signal

## Limitations & Future Work

### Current Limitations
1. **External Events**: Model doesn't capture unforeseen events (e.g., extreme weather, local events)
2. **New Stores**: Limited data for recently opened stores
3. **Anomaly Detection**: Outlier sales days may not be well-predicted
4. **Feature Interactions**: Some complex 3-way interactions may be missed

### Potential Improvements
1. **Advanced Features**:
   - Weather data integration
   - Local economic indicators
   - Marketing campaign details
   - Product-level granularity

2. **Model Enhancements**:
   - Ensemble methods (stacking XGBoost with other models)
   - Deep learning approaches (LSTM, Transformer for time series)
   - Bayesian optimization for hyperparameter tuning
   - Multi-output predictions (simultaneous sales and customer forecasting)

3. **Validation**:
   - Walk-forward validation for time series
   - Store-level cross-validation
   - Stratified sampling by store characteristics

4. **Production Deployment**:
   - Real-time prediction API
   - Automated retraining pipeline
   - Monitoring and alerting system
   - A/B testing framework

## Technical Notes

### Memory Considerations
- Model size: 351MB (compressed JSON)
- Training dataset: ~1M rows requires ~500MB RAM
- Feature engineering increases memory footprint
- Recommend 8GB+ RAM for smooth execution

### Processing Time
- Training: ~10-30 minutes (depending on hardware)
- Prediction: ~1-2 seconds for full test set
- Feature engineering: ~30 seconds

### Reproducibility
- Random seed management in XGBoost
- Deterministic preprocessing pipeline
- Versioned dependencies recommended

## References

- **Kaggle Competition**: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- **XGBoost Documentation**: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- **Original Challenge**: Rossmann 2015 Kaggle Competition

## License

This project is created for educational and analytical purposes as part of the Rossmann Store Sales Kaggle Challenge.

## Contact

For questions or collaboration opportunities, please refer to the project repository.

---

**Last Updated**: January 2026
**Model Version**: PLATINUM (RMSPE: 12.49%)
**Total Forecasted Sales**: €243.6M (6-week period)
