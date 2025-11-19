# Quick Setup Guide

## Setup

1. Clone repo and navigate to directory
2. Create virtual environment: `python3 -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`
5. Copy `.keras` folder to `C:\Users\[YourUsername]\`
6. Place JPG images in `db/` directory

## Run

```sh
streamlit run main.py
```

Open `http://localhost:8501`

## Troubleshooting

- Memory issues: Reduce database size
- Slow startup: Normal on first run
- Model errors: Check `.keras` folder location
- Format errors: Use JPG only

See `README.md` for details.
