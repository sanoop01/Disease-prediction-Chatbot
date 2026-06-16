# Disease Prediction Chatbot - Setup Steps

## Step 1: Create Project Folder

```bash
mkdir Disease_Chatbot
cd Disease_Chatbot
```

## Step 2: Create Files

Create:

- app.py
- README.md
- requirements.txt

## Step 3: Install Dependencies

```bash
python -m pip install --user streamlit pandas scikit-learn joblib numpy
```

## Step 4: Verify Streamlit

```bash
python -m pip show streamlit
```

## Step 5: Run Application

```bash
python -m streamlit run app.py
```

## Step 6: Open Browser

Visit:

http://localhost:8501

## Step 7: Use the Application

1. Upload CSV dataset
2. Select target column
3. Train model
4. Enter patient values
5. Predict disease outcome

## Troubleshooting

If 'streamlit' is not recognized:

```bash
python -m streamlit run app.py
```

Check version:

```bash
python -m streamlit --version
```
