from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE/"data/customer_churn.csv"
OUT = BASE/"outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nFirst 5 rows:\n", df.head())

# EDA
df["churn"].value_counts().plot(kind="bar", title="Customer Churn Distribution")
plt.xlabel("Churn"); plt.ylabel("Customers"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(OUT/"churn_distribution.png", dpi=150); plt.close()

rates = df.groupby("contract")["churn"].apply(lambda x:(x=="Yes").mean()*100).sort_values(ascending=False)
rates.plot(kind="bar", title="Churn Rate by Contract Type")
plt.xlabel("Contract"); plt.ylabel("Churn Rate (%)"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(OUT/"churn_by_contract.png", dpi=150); plt.close()

X = df.drop(columns=["customer_id","churn"])
y = (df["churn"]=="Yes").astype(int)

num = ["tenure_months","monthly_charges","support_tickets"]
cat = ["contract","internet_service","payment_method","tech_support","senior_citizen","dependents"]

pre = ColumnTransformer([
    ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),
    ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)

models={
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "Random Forest":RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced")
}
results=[]
for name,model in models.items():
    pipe=Pipeline([("preprocessor",pre),("model",model)])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    results.append({
        "Model":name,
        "Accuracy":accuracy_score(y_test,pred),
        "Precision":precision_score(y_test,pred,zero_division=0),
        "Recall":recall_score(y_test,pred,zero_division=0),
        "F1 Score":f1_score(y_test,pred,zero_division=0)
    })
    print("\n"+"="*55); print(name); print("="*55)
    print(classification_report(y_test,pred,target_names=["Stayed","Churned"],zero_division=0))
    ConfusionMatrixDisplay(confusion_matrix(y_test,pred),display_labels=["Stayed","Churned"]).plot()
    plt.title(f"Confusion Matrix - {name}"); plt.tight_layout()
    plt.savefig(OUT/f"confusion_matrix_{name.lower().replace(' ','_')}.png",dpi=150); plt.close()

results_df=pd.DataFrame(results).sort_values("F1 Score",ascending=False)
results_df.to_csv(OUT/"model_results.csv",index=False)
print("\nMODEL COMPARISON\n",results_df.to_string(index=False))

best_name=results_df.iloc[0]["Model"]
best=Pipeline([("preprocessor",pre),("model",models[best_name])])
best.fit(X_train,y_train)
pred=best.predict(X_test)
proba=best.predict_proba(X_test)[:,1]
out=X_test.copy()
out["actual_churn"]=y_test.values
out["predicted_churn"]=pred
out["churn_probability"]=proba.round(3)
out.to_csv(OUT/"customer_churn_predictions.csv",index=False)
print("\nBest model:",best_name)
print("Project completed. Check the outputs folder.")
