# Reddit Post Moderation Classifier 🤖

This project is a complete, end-to-end data science pipeline designed to analyze and predict the moderation status of Reddit posts. It performs comprehensive data analysis, feature engineering, model training, and integrates with the Gemini API for advanced, AI-driven moderation decisions.

---
## ✨ Features

- **Comprehensive EDA**: Performs SQL-style analysis and correlation checks on the dataset.
- **Data Visualization**: Generates plots for score distributions, posting patterns, feature correlations, and word clouds.
- **NLP Preprocessing**: Cleans and preprocesses text data using tokenization, stop-word removal, and stemming.
- **Feature Engineering**: Creates a rich feature set including TF-IDF for text and various numerical features.
- **Classic ML Modeling**: Trains and evaluates Decision Tree and Random Forest classifiers.
- **Gemini API Integration**: Makes a live API call to a Google Gemini model for a moderation decision based on a structured JSON schema.
- **Modular & Scalable**: The entire project is built with a clean, modular structure, with all settings managed through configuration objects.

---
## 📂 Project Structure

The project is organized into a modular structure to separate concerns and improve maintainability.

reddit_moderation_project/
├── data/
│   └── r_dataisbeautiful_posts.csv
├── src/
│   ├── init.py
│   ├── analysis.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── gemini_integration.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── reporting.py
│   ├── utils.py
│   └── visualization.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt


---
## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/Anshu404/reddit-moderation-pipeline.git](https://github.com/Anshu404/reddit-moderation-pipeline.git)
cd reddit-moderation-pipeline
2. Create and Activate a Virtual Environment
This project uses a virtual environment to manage dependencies.

Bash

# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate
3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

Bash

pip install -r requirements.txt
4. Set Up Your Gemini API Key
This project requires a Google Gemini API key to run Step 8.

Get your key from Google AI Studio.

Set the key as an environment variable in your terminal. You must do this every time you open a new terminal session.

Bash

export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
▶️ How to Run
With your virtual environment active and the API key set, you can run the entire pipeline with a single command.

Bash

python3 main.py --data data/r_dataisbeautiful_posts.csv
For a faster run on a smaller subset of the data, use the --sample flag:

Bash

python3 main.py --data data/r_dataisbeautiful_posts.csv --sample 5000
🚀 Next Steps & Future Improvements
The current models serve as a baseline. The next phase of this project will focus on improving model performance by:

Handling Class Imbalance: Implementing techniques like SMOTE or using class_weight='balanced' in the models.

Hyperparameter Tuning: Using GridSearchCV to find the optimal settings for the classifiers.

Trying Advanced Models: Experimenting with LogisticRegression, XGBoost, and LightGBM.

Advanced Feature Engineering: Exploring word embeddings (e.g.,
