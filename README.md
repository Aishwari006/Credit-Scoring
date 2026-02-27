# LiquidityLens AI

## Cash-Flow Based MSME Underwriting Intelligence Platform

---

## Overview

LiquidityLens AI is an advanced cash-flow-based credit assessment platform designed for MSME (Micro, Small, and Medium Enterprises) underwriting.

Traditional credit scoring models often fail MSMEs due to limited formal credit history. LiquidityLens AI addresses this gap by ingesting raw bank transaction data, applying a machine learning behavior engine, and utilizing an agentic LLM to generate explainable alternative credit scores.

The system focuses purely on financial behavior derived from transactional cash flow rather than traditional bureau data.

---

## Key Features

### Alternative Credit Scoring

* Uses a Random Forest machine learning model
* Computes Probability of Default (PD)
* Generates an alternative credit score (0–900)
* Based entirely on cash flow behavior

### Agentic Explainability Engine

* Combines deterministic financial rules with an LLM (Ollama / Qwen)
* Produces human-readable underwriting summaries
* Identifies specific risk drivers such as:

  * High cash burn
  * Counterparty concentration risk
  * Liquidity instability

### Interactive Analytics Dashboard

* Visualizes:

  * Liquidity stability
  * Operational velocity
  * Monthly inflow vs outflow
  * Categorized expense structure
* Built using Chart.js

### Portfolio Command Center

* Central dashboard for underwriters
* Displays:

  * MSME credit ledger
  * Approval / rejection metrics
  * Risk distribution
* Enables deep-dive into individual company profiles

### Account Aggregator Simulation

* Demonstrates RBI-compliant 7-step cryptographic data flow
* Simulates secure consent-based data ingestion architecture

### Secure Authentication

* Built using Django session-based authentication
* Role-based access control for underwriters

---

## Tech Stack

### Backend

* Python
* Django
* Pandas
* Scikit-Learn (Random Forest)

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* Chart.js

### Database

* SQLite (Development)

### AI / LLM Integration

* Ollama (Qwen3:8b) for local inference
* Groq API (optional cloud deployment)

---

## Prerequisites

Before running this project, ensure you have installed:

* Python 3.9 or higher
* Git
* Ollama (only required for local LLM inference)

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/liquiditylens-ai.git
cd liquiditylens-ai
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
```

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize the Database

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create an Admin / Underwriter Account

```bash
python manage.py createsuperuser
```

Follow the prompts to set up your username and password.

### 6. Start the Local AI Model (Optional for Explainability)

Open a separate terminal and run:

```bash
ollama serve
```

Then load the model:

```bash
ollama run qwen3:8b
```

If deploying to the cloud, modify `explainability.py` to use a cloud LLM provider such as Groq or OpenAI instead of Ollama.

### 7. Run the Application

```bash
python manage.py runserver
```

---

## Usage Guide

1. Navigate to:

   [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

2. Log in using your superuser credentials.

3. On the Data Ingestion page:

   * Enter Business Name
   * Enter GSTIN
   * Enter PAN (10 characters, e.g., ABCDE1234F)
   * Upload a synthetic bank statement CSV

4. The system will:

   * Process transactions
   * Execute ML underwriting pipeline
   * Generate Probability of Default
   * Produce alternative credit score
   * Generate LLM explainability summary
   * Redirect to Deep Dive Dashboard

5. Click "Portfolio Command Center" to:

   * View aggregate MSME ledger
   * Track approvals, rejections, reviews
   * Analyze individual company risk

---

## Project Structure

```
/uploads/
    views.py
    forms.py
    models.py
    Handles file ingestion and routing

/Credit-Scoring/
    underwrite.py
    behaviour_engine.py
    explainability.py
    Core ML logic and LLM integration

/templates/
    login.html
    upload.html
    dashboard.html
    portfolio.html
    Frontend interface
```

---

## Architecture Summary

LiquidityLens AI follows a layered underwriting architecture:

1. Data Ingestion Layer
2. Behavioral Feature Engineering Engine
3. ML Risk Model (Random Forest)
4. Deterministic Rule Layer
5. LLM Explainability Agent
6. Portfolio Aggregation Layer

This design enables scalable, transparent, and alternative credit underwriting for MSMEs.
