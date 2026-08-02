# 👶 Missing Child Identification System

An AI-powered web application designed to assist in locating and cross-matching missing children using facial recognition, deep learning embeddings, and real-time database queries. Built with **Streamlit**, **DeepFace (FaceNet)**, and **SQLite**, this application empowers both citizens ("Good Samaritans") and law enforcement to report, search, and manage missing child records efficiently.

---

## ✨ Key Features

* **📌 Case Registration:** Securely register new missing or found child reports with personal details, estimated age, location, and high-resolution reference photographs.
* **🔍 AI Facial Recognition & Cross-Matching:** Instant deep learning facial search powered by **FaceNet** (512-D embeddings) using cosine similarity scoring against registered records.
* **📷 Dual Input Options:** Real-time webcam scanning or file upload for immediate field-sighting verification.
* **📋 Public Directory & Card Gallery:** Citizen-facing searchable directory to view registered missing children and verified sighting audit logs.
* **🔒 Secured Admin Controls:** Role-based access protection (Password: `admin123`) for administrative tasks like deleting records, clearing logs, or resetting system data.
* **🏥 Child Welfare & Emergency Locator:** Direct emergency contacts (**1098 Childline / 112 Police**) and interactive directory of official Child Welfare Committees (CWC) and shelters.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **AI & Computer Vision:** [DeepFace](https://github.com/serengil/deepface) (FaceNet Model), OpenCV-Headless, PIL (Pillow)
* **Database:** SQLite3
* **Data Processing:** Pandas, NumPy

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies for local & cloud setup
├── README.md                   # Project documentation
├── registered_children/        # Storage folder for registered reference images
└── spotted_children/           # Storage folder for sighting verification images

---

## 🚀 How to Run Code

# Open your Terminal / Command Prompt and navigate to your project directory:
cd "MISSING CHILD IDENTIFICATION SYSTEM PROJECT"

# Run the Streamlit application:
streamlit run app.py

---