#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Hello")


# In[16]:


import sagemaker
import boto3

bucketname = 'heartdisease-data'
myregion = boto3.session.Session().region_name
print(myregion)


# In[17]:


s3 = boto3.resource('s3')
try:
    if myregion == 'ap-south-1':
        s3.create_bucket(Bucket=bucketname, CreateBucketConfiguration={'LocationConstraint': myregion})
    else:
        s3.create_bucket(Bucket=bucketname)
    print('S3 bucket successfully created')
except Exception as e:
    print('S3 error:', e)


# In[46]:


import boto3
import pandas as pd

# Initialize S3 client
s3 = boto3.client('s3')

# Specify bucket name and file name
bucket_name = 'heartdisease-data'
file_name = 'heart.csv'

# Load CSV file from S3 into pandas DataFrame
obj = s3.get_object(Bucket=bucket_name, Key=file_name)
df = pd.read_csv(obj['Body'])
print("Column names:")
print(df.columns)


# In[47]:


import pandas as pd
from sklearn.model_selection import train_test_split
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# In[48]:


import boto3
from io import StringIO

# Initialize S3 client
s3 = boto3.resource('s3')

# Define bucket and file paths
bucket_name = 'heartdisease-data'
train_data_key = 'heartdisease/train/train.csv'
test_data_key = 'heartd/test/test.csv'

# Convert DataFrame to CSV format (adjust if needed)
csv_buffer = StringIO()
X_train.to_csv(csv_buffer, index=False)
s3.Object(bucket_name, train_data_key).put(Body=csv_buffer.getvalue())

csv_buffer = StringIO()
X_test.to_csv(csv_buffer, index=False)
s3.Object(bucket_name, test_data_key).put(Body=csv_buffer.getvalue())

print(f"Training data uploaded to s3://{bucket_name}/{train_data_key}")
print(f"Testing data uploaded to s3://{bucket_name}/{test_data_key}")


# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Apply transformers to columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Define the model with preprocessing steps
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


# In[37]:


from sagemaker import get_execution_role
from sagemaker import LinearLearner

# SageMaker role
role = get_execution_role()

# Define the Linear Learner estimator
linear = LinearLearner(role=role,
                       train_instance_count=1,
                       train_instance_type='ml.m4.xlarge',
                       predictor_type='binary_classifier',  # Adjust for your problem type
                       output_path=f's3://{bucket_name}/linear-learner/output')


# In[40]:





# In[ ]:




