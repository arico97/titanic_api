import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

class Datanalysizer:
    def __init__(self, train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path 
            
    def read_data_encode(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        df = pd.concat([train,test])
        df['Sex'] = pd.to_numeric(df['Sex'],downcast='integer')
        df['Age'] = pd.to_numeric(pd.cut(df['Age'],bins = [0,18,32,48,64,100], labels = ['0','1','2','3','4']),downcast='integer')
        df.drop(['Cabin','Name','Ticket','Fare'], axis = 1, inplace = True)
        df['Embarked'].replace({'S':'0','Q':'1','C':'2'},inplace = True)
        df['Embarked'] = pd.to_numeric(df['Embarked'],downcast='integer')
        return df 

    def make_plots(self,df):
        df.hist(figsize=(12, 8))
        plt.savefig('./figures/histogram.png')
        plt.close()

        plt.scatter_matrix(df)
        plt.savefig('./figures/scatter_matrix.png')
        plt.close()

        g = sns.FacetGrid(df, col='Survived')
        g.map(plt.hist, 'Age', bins=20)
        g.figure.savefig('./figures/surv/age_surv.png')
        plt.close(g)

        grid = sns.FacetGrid(df, col='Survived', row='Pclass',  aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.6, bins=15)
        grid.add_legend()
        grid.figure.savefig('./figures/surv/pclass_age_surv.png')
        plt.close(grid)

        grid = sns.FacetGrid(df, col='Survived', row='Sex', aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.6, bins=15)
        grid.add_legend()
        grid.figure.savefig('./figures/surv/sex_age_surv.png')
        plt.close(grid)
        
        grid = sns.FacetGrid(df, row='Embarked', aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
        grid.add_legend()
        grid.figure.savefig('./figures/surv/emb_sex_pclass_surv.png')
        plt.close(grid)
        # return json with the figures path
        