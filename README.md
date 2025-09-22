# 📌 Project Overview – SVM Classifier with Streamlit & Docker

This project demonstrates how to build, containerize, and deploy a **machine learning web application** using **Support Vector Machines (SVM)**, **Streamlit**, and **Docker**.

### 🔹 What the App Does

* Generates **synthetic classification data** using `sklearn.datasets.make_classification`.
* Trains an **SVM model** with **GridSearchCV** for hyperparameter tuning.
* Provides an **interactive Streamlit interface** where users can:

  * Manually input feature values and get **real-time predictions**.
  * Visualize the dataset with **scatter plots**.
  * View **model performance metrics** including accuracy, confusion matrix, and classification report.

### 🔹 Tech Stack

* **Python 3.11**
* **Libraries**: scikit-learn, pandas, numpy, seaborn, matplotlib, streamlit
* **Docker**: For containerization and portability

### 🔹 Why Docker?

* Ensures a **reproducible environment**
* Runs seamlessly on any machine (Windows, Linux, macOS, servers)
* Easy to share via **Docker Hub**
* One command deploys the full ML web app

### 🔹 How to Run

```bash
# Build image
docker build -t svm-streamlit .

# Run container
docker run -p 8501:8501 svm-streamlit
```

Then open 👉 [http://localhost:8501](http://localhost:8501) in your browser.

### 🔹 Deployment

The image can be pushed to **Docker Hub** and shared publicly.
Example:

```bash
docker tag svm-streamlit your-dockerhub-username/svm-streamlit:latest
docker push your-dockerhub-username/svm-streamlit:latest
```

---

✨ With this project, you now have a complete workflow:
**ML model → Streamlit app → Docker container → Share/Deploy anywhere.**

