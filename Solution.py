import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold

from xgboost import XGBClassifier

def load_and_merge_dataset():
  loan= pd.read_csv("loan/loan_table.csv")
  borrower= pd.read_csv("loan/borrower_table.csv")
  print("Loan data has {} rows and {} columns".format(loan.shape[0],loan.shape[1]))
  print("Borrower data has {} rows and {} columns".format(borrower.shape[0],borrower.shape[1]))
  final=pd.merge(loan, borrower, on="loan_id")
  print("Final data has {} rows and {} columns".format(final.shape[0],final.shape[1]))
  return numeric_conversion_missing_data(final)



def numeric_conversion_missing_data(final):
  final['loan_purpose']= final['loan_purpose'].astype('category')
  final['purpose'] = final['loan_purpose'].cat.codes
  final['day']= final.apply(lambda x: int(x.date.split('-')[-1]), axis=1)
  final['month']= final.apply(lambda x: int(x.date.split('-')[1]), axis=1)
  final['year']= final.apply(lambda x: int(x.date.split('-')[0]), axis=1)
  final.drop(columns=['year'], inplace=True)
  final['currently_repaying_other_loans'].fillna(value=0, inplace=True)
  final['fully_repaid_previous_loans'].fillna(value=1.0, inplace=True)
  return final


def categorical_continous_scaling(final, features, number_of_categorical):
  granted= final[final['loan_granted']==1]
  not_granted= final[final['loan_granted']==0]
  # Define the numerical and categorical columns
  num_cols = features[number_of_categorical:]
  cat_cols = features[:number_of_categorical]
  # Build the column transformer and transform the dataframe
  col_trans = ColumnTransformer([
      ('num', MinMaxScaler(), num_cols),
      ('cat', OneHotEncoder(drop='if_binary'), cat_cols)
  ])
  granted_transformed = col_trans.fit_transform(granted)

  X= granted_transformed
  y= granted[['loan_repaid']]
  # Stratified split of the train and test set with train-test ratio of 7:3
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                              stratify=y, random_state=10)
  return X_train, X_test, y_train, y_test 

def feature_engineering(final):
  final['cred']=1
  final.loc[final['avg_percentage_credit_card_limit_used_last_year'].isna()==True,'cred']=0
  final['avg_percentage_credit_card_limit_used_last_year'].fillna(final['avg_percentage_credit_card_limit_used_last_year'].mean(),inplace=True)
  final['is_emergency']=0
  final.loc[final['loan_purpose']=='emergency_funds','is_emergency']=1
  return final



def final_model(final):
  features= ['currently_repaying_other_loans',
             'is_employed',
             'is_emergency',
             'fully_repaid_previous_loans',
             'cred',
             'avg_percentage_credit_card_limit_used_last_year',
             'total_credit_card_limit',
               'saving_amount',
             'checking_amount',
               'yearly_salary',
             'age',
             'dependent_number']
  out= " ,".join(features)
  print("Final featureset-: ", out)
  model = XGBClassifier(max_depth=3,eta=0.1, n_estimators=150)
  #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  #n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  #print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
  X_train, X_test, y_train, y_test= categorical_continous_scaling(final, features, 5)
  model.fit(X_train, y_train.values.ravel())
  y_pred=model.predict(X_test)
  print('*'*10)
  print('F1 Score is:')
  print(f1_score(y_pred, y_test))
  print('*'*10)
  print('Accuracy Score is:')
  print(accuracy_score(y_pred,y_test))
  print('*'*10)
  print('ROC AUC Score is:')
  print(roc_auc_score(y_pred,y_test))


if __name__ == "__main__":
  final= load_and_merge_dataset()
  print("Engineering Features")
  final= feature_engineering(final)
  print("Training Model")
  final_model(final)
