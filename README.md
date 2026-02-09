# mask-detect
Face mask detection using CNN.

## Streamlit Community Cloud deployment

1. Set Python version to `3.11` (or `3.12`) in app **Advanced settings**.
2. Redeploy after updating settings.

Notes:
- TensorFlow inference is disabled automatically on Python `3.13+` to avoid runtime crashes.
- `requirements.txt` installs TensorFlow only for Python versions below `3.13`.
