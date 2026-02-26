# Cash-Flow Based Credit Assessment for Thin-File SMEs

This project is a Django-based prototype built for the **FinCode Hackathon**, focusing on evaluating SME creditworthiness using cash-flow data instead of traditional credit history or collateral.

The current version includes a clean borrower-facing UI for:

- Entering basic business details
- Uploading financial records (CSV / Excel)
- Securely storing data using SQLite

---

## Tech Stack

- Python 3.10+
- Django
- SQLite (default Django database)
- Bootstrap (UI)

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone <repo-url>
cd Credit-Scoring


2. Create and activate a virtual environment
Windows

python -m venv venv
venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate



3. Install dependencies
pip install -r fincode/requirements.txt


4. Run database migrations
cd fincode
python manage.py migrate


5. Start the development server
python manage.py runserver
Open your browser and go to:

http://127.0.0.1:8000/
```
