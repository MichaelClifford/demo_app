#import operator
import numpy as np
import pandas as pd
import joblib
import boto3
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve,\
                            average_precision_score,\
                            roc_auc_score, roc_curve,\
                            confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE



## Connect to s3

my_bucket = "MICHAEL-DATA-ODSC"
conn = boto3.client(service_name='s3',
        aws_access_key_id= os.environ.get("AWS_ACCESS_KEY_ID")  ,
        aws_secret_access_key= os.environ.get("AWS_SECRET_ACCESS_KEY"),
        endpoint_url= os.environ.get("S3_ENDPOINT_URL"))

# Read in Data from S3

obj = conn.get_object(Bucket=my_bucket, Key='demo-data/creditcard.csv')
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

# Pre-process

# Train a model
df_train, df_test = train_test_split(df, train_size=0.75)
model = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=10)
model.fit(df_train.drop(['Time', 'Class'], axis=1),df_train['Class'])
test_pred = model.predict(df_test.drop(['Time', 'Class'] ,axis=1))
test_label = df_test['Class']
test_acc = np.sum(test_pred==test_label) / len(test_pred)
print(f'test_acc = {test_acc}')

print("saving model to S3 storage")

# Save model
joblib.dump(model, 'model.pkl')

# Upload to S3
my_bucket = "MICHAEL-DATA-ODSC"
key = "models/model.pkl"
file  = "model.pkl"

conn.upload_file(Bucket=my_bucket, Key=key, Filename=file)
conn.list_objects(Bucket=my_bucket)
