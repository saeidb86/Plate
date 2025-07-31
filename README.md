نصب محیط مجازی و کتابخانه‌ها:

powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt


نصب محیط:

bash
pip install -r requirements.txt
اجرای سرویس:

bash
python app.py
تست با curl:

bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/recognize




{
  "results": [
    {
      "text": "۱۲۳۴۵۶۷۸",
      "bbox": [100, 150, 300, 200],
      "confidence": 0.95
    }
  ]
}