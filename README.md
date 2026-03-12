# Kidney Stone Detection System with Explainable AI

This is a complete AI-Powered Kidney Stone Detection System using Clinical Data (SVM) and CT Scan Images (CNN) with Grad-CAM explainability, wrapped in a professional Flask web application.

## 📂 Project Structure

```
KidneyStoneProject/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── instance/
│   └── database.db         # SQLite database (Users, History)
├── static/
│   ├── css/                # Stylesheets
│   ├── uploads/            # Uploaded CT scans
│   └── heatmaps/           # Generated Grad-CAM heatmaps
├── templates/              # HTML Templates (Login, Clinical, CT, Reports)
├── utils/                  # Utility scripts (Grad-CAM, Severity Logic)
├── clinical_svm_model.pkl  # Trained SVM Model
├── scaler.pkl              # Data Scaler
└── ct_vgg16_stone_model.h5 # Trained VGG16 CNN Model
```

## 🚀 Installation Steps

1.  **Install Python**: Ensure Python 3.8+ is installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Verify Models**: Ensure `clinical_svm_model.pkl`, `scaler.pkl`, and `ct_vgg16_stone_model.h5` are in the project root.

## 🏃 parameters to Run the Application

1.  Open a terminal in the project directory.
2.  Run the application:
    ```bash
    python app.py
    ```
3.  Open your browser and navigate to:
    `http://127.0.0.1:5000`

## 🔑 Key Features & Usage

### 1. User Authentication
- **Register**: Create a new account to save your history.
- **Login**: Secure access to prediction tools.

### 2. Clinical Data Analysis
- Enter patient's urine analysis parameters (Specific Gravity, pH, etc.).
- The system predicts **"Kidney Stone Detected"** or **"No Stone"**.
- Establishes a **Severity Level** (Low, Medium, High).
- Provides natural dietary suggestions based on severity.

### 3. CT Scan Analysis
- Upload a CT Scan image (JPG/PNG).
- The Deep Learning model (VGG16) analyzes the image.
- **Grad-CAM Heatmap**: Visualizes the exact region where the stone is detected.
- **Confidence Score**: Displays the model's certainty.

### 4. History & Reports
- **My History**: View all past predictions.
- **Formal Report**: Generate a printable, professional medical report with all details and AI explanations.

## 🌐 Deployment Logic

### Local Deployment
- Simply run `python app.py`.

### Cloud Deployment (Render/Heroku)
1.  Push code to GitHub.
2.  Link repository to Render/Heroku.
3.  Set build command: `pip install -r requirements.txt`.
4.  Set start command: `gunicorn app:app`.

## 🛠 Database
- Uses **SQLite** via **SQLAlchemy**.
- `User` table stores credentials.
- `History` table stores prediction logs, timestamps, and results.
- Database is automatically created on first run in the `instance/` folder.
