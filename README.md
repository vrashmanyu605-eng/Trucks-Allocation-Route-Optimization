# Fleet Allocation Optimization System

A production-ready implementation for vehicle-to-delivery-order (DO) allocation, designed to maximize profit while satisfying operational constraints.

## Overview

This system automates the process of assigning available vehicles to pending delivery orders. It uses a combination of machine learning (Random Forest) for profitability prediction and linear programming (PuLP/CBC) for optimal allocation.

## Key Features

-   **Data Preparation Layer**: Fetches and cleans data from CSV files (DO Dump, LR Details, Vehicle Master, Route Master).
-   **Data Enrichment**: Calculates weights from dimensions, fetches GPS locations, and estimates service schedules.
-   **Constraint Validation**: Enforces hard constraints including:
    -   Vehicle capacity vs. load weight.
    -   Vehicle type compatibility.
    -   Service maintenance intervals.
    -   Delivery deadline feasibility based on estimated travel time.
    -   Geographic permit validity.
-   **Profitability Prediction**: A Random Forest model trained on historical LR data to predict the profit margin for specific vehicle-route combinations.
-   **Optimization Engine**: Uses the PuLP library to solve the assignment problem, maximizing total profit while ensuring each vehicle and DO are uniquely assigned.
-   **Market Hire Recommendation**: Automatically identifies DOs that cannot be fulfilled by the internal fleet and recommends market hiring.

## Project Structure

-   [`test.py`](test.py): The main script containing the entire system logic.
-   `1_DO_Dump_Enhanced_1000.csv`: Delivery order data.
-   `2_LR_Details_Enhanced_1000.csv`: Historical trip/profit data for model training.
-   `3_Vehicle_Master.csv`: Fleet information and availability.
-   `4_Route_Master.csv`: Distance and backhaul probability data for routes.
-   `requirements.txt`: Python dependencies.

## Installation

1.  Ensure you have Python 3.x installed.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main orchestrator script:

```bash
python test.py
```

The system will:
1.  Train/Load the profitability model.
2.  Fetch and enrich pending DO and vehicle data.
3.  Run the optimization solver.
4.  Generate a summary in the console.
5.  Export results to `daily_allocations.csv` and `daily_allocations.json`.

## Technical Stack

-   **Language**: Python
-   **Data Handling**: `pandas`, `numpy`
-   **Machine Learning**: `scikit-learn`, `joblib`
-   **Optimization**: `pulp` (with CBC solver)
-   **Logging**: Python standard `logging` library
