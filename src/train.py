import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class MakeTitanicPrediction:
    def __init__(self, train_path,test_path):
            self.train_path = train_path
            self.test_path = test_path 
            
    def preprocess_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        df = pd.concat([train,test])
        df.replace({'male':'0','female':'1'},inplace=True) 
        df['Sex'] = pd.to_numeric(df['Sex'],downcast='integer')
        df['Age'] = pd.to_numeric(pd.cut(df['Age'],bins = [0,18,32,48,64,100], labels = ['0','1','2','3','4']),downcast='integer')
        df.drop(['Cabin','Name','Ticket','Fare'], axis = 1, inplace = True)
        df['Embarked'].replace({'S':'0','Q':'1','C':'2'},inplace = True)
        df['Embarked'] = pd.to_numeric(df['Embarked'],downcast='integer')
        regression_imputer = IterativeImputer(estimator=RandomForestClassifier())
        lr_imp = regression_imputer.fit(np.array(df['Pclass']).reshape(-1,1),np.array(df['Age']).reshape(-1,1) )
        impt=lr_imp.transform(df.iloc[:,[3]].values)
        df['Age_imputated']=impt
        df['Age_imputated']=round(df['Age_imputated'])
        df.drop('Age',axis=1,inplace=True)
        return df,len(train)
    

    def split_val(self,dtf,leng):    
        train_, test_ = dtf.iloc[:leng,:],dtf.iloc[leng:,:]
        Y_train = train_['Survived']
        X_train = train_.loc[:,dtf.columns != 'Survived']
        X_test= test_.loc[:,dtf.columns != 'Survived']
        return X_train, X_test, Y_train
    

    def make_prediction(self,X_train, X_test, Y_train):
        model = RandomForestClassifier()
        model.fit(X_train,Y_train)
        prediction = model.predict(X_test)
        return prediction

    def path_out_write(self):
        data_prep,lent=self.preprocess_data() 
        X_train, X_test, Y_train = self.split_val(data_prep,lent)
        prediction = self.make_prediction(X_train, X_test, Y_train) 
        out_path = self.test_path.replace("test","predictions") 
        prediction_df=pd.DataFrame(prediction)
        prediction_df.to_csv(out_path,sep="|",index=False)
        return out_path
        
         
    
