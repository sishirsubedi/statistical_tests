import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import seaborn as sns
import random
from statsmodels.graphics.factorplots import interaction_plot
import rpy2.robjects as robjects
from sklearn.feature_selection import f_regression
from scipy.stats import f
#fisher exact test

'''
        School1  School2
male      8        12
female    12        8
'''
gender = [[7,3],[3,7]]
oddsratio, pvalue = stats.fisher_exact(gender)
print (pvalue)


#chi square test
''' 	Republican 	Democrat 	Totals
M 	    215 	    143 	    358
F 	    19      	64 	        83
Totals 	234 	    207 	    441'''

house = [ [ 314, 44 ], [  33,334 ] ]
chi2, p, ddof, expected = stats.chi2_contingency( house )
print (p)

diceroll1 = [9,10,12,11,8,10]
diceroll2 = [6,0,14,20,11,9]
dices =[diceroll1,diceroll2]
chi2, p = stats.chisquare( diceroll2 )
print (p)
chi2, p, ddof, expected = stats.chi2_contingency(dices)
print (p)



### chit test for one way table -- phenotype 9:3:3:1 ratio
obs = [284,21,21,55]
exp = [214.3,71.4,71.4,23.8]
phenotype = [obs,exp]
chi2, p, ddof, expected = stats.chi2_contingency(phenotype)
print (p)

### chi test for two way table ( test for independence)
### drinking problem
       # low, mid, high times
police =[71,154,398]
nopolice =[4992,2808,2737]
drinking = [police,nopolice]
chi2, p, ddof, expected = stats.chi2_contingency(phenotype)
print (p)
### here important expectation is calculated as : (rowsum * columnsum)/(total)



'''
There are three main types of t-test:
1. An Independent Samples t-test compares the means for two groups.
2. A Paired sample t-test compares means from the same group 
   at different times (say, one year apart).
3. A One sample t-test tests the mean of a single group against a known mean.
'''

## one sample t test with population variance uknown and estimated using sample variance
N = 1000
notnormal = np.random.rand(N)
np.mean(notnormal)
sm.qqplot(notnormal)
sns.boxplot(y=notnormal)

group1 = np.array([177.3, 182.7, 169.6, 176.3, 180.3, 179.4, 178.5, 177.2, 181.8, 176.5])
sm.qqplot(group1)
sns.boxplot(y=group1)
print (np.mean(group1))

pop_mean = 176
print (stats.ttest_1samp(group1,pop_mean))


## unpaired t test for independent mean comparison
female = [63.8, 56.4, 55.2, 58.5, 64.0, 51.6, 54.6, 71.0]
male = [75.5, 83.9, 75.7, 72.5, 56.2, 73.4, 67.7, 87.9]

gender = pd.DataFrame({'male':male, 'female':female})
sns.boxplot(data=gender)
two_sample = stats.ttest_ind(male, female)
print (two_sample)

# assuming unequal population variances
two_sample_diff_var = stats.ttest_ind(male, female, equal_var=False)
print (two_sample_diff_var)


### paired t test
## idea is take difference of condition1 and condition2, and use this difference value to
## to do 1 sample t test to check if with population mean assumption as zero
## pain score before and after surgery
before = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
after = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]
painscore = pd.DataFrame({'before':before, 'after':after})
sns.boxplot(data=painscore)
paired_sample = stats.ttest_rel(before, after)
print  (paired_sample)



############ test for variance

##### 1. test for population variance from a single sample -- chi square test
## alchohol percentage in white wine, population variance is 0.5 and we want to check if our sample
## has more alcohol percentage

samples = [4.68 ,4.13, 4.80, 4.63, 5.08, 5.79 ,6.29 ,6.79,4.93 ,4.25, 5.70 ,4.74 ,5.88 ,6.77, 6.04, 4.95]
sample_numbers = len(samples)
df = sample_numbers - 1
sample_mean = np.mean(samples)
sample_std = np.std(samples)
sample_variance = np.var(samples)
population_variance = 0.5

chival = ((sample_numbers -1)*sample_variance)/population_variance
print (chival)
### one tailed test on significance level of 0.05-- variance is greater than population
siglevel = 0.05
print (stats.chi2.ppf(1- siglevel, df=df))

### since our chival(20) is smaller than significant level value(24.9)-- we cant reject null hypothesis
### that is these two variance are same
## sample variance of 0.67 is not greater than population variance 0.5



## f test for two population variance -- try r function in PYTHON using rpy2
##good explanation : https://www.youtube.com/watch?v=Pml68e3Eh3o
# sample with higher variance  is on numerator in F test
#

def Ftest_pvalue_rpy2(d1,d2):
    """docstring for Ftest_pvalue_rpy2"""
    rd1 = (robjects.FloatVector(d1))
    rd2 = (robjects.FloatVector(d2))
    rvtest = robjects.r['var.test']
    return rvtest(rd1,rd2)

company1 = [860,850,750,870,740,410,410,820,890,890]
company2=[540,640,600,640,300,610,430,280,300,610]

print (Ftest_pvalue_rpy2(company1,company2))


company1 = [1,1,1,1,1,2]
company2=[1,1,1,1,1,2]
print (Ftest_pvalue_rpy2(company1,company2))



########### one way anova - comparing three or more sample means
##same ideas as testing population variance between two population
##https://www.youtube.com/watch?v=UrRYITjDOww
## F = MS_Columns/ MS_Within_Errors
# MS_Columns = SS_Columns/Columns-1
# MS_Within_Errors = N_Columns


tech1 = [random.randint(25,50) for x in range(0,100)]
tech2 = [random.randint(45,70) for x in range(0,100)]
tech3 = [random.randint(25,100) for x in range(0,100)]

techs = pd.DataFrame({'t1':tech1, 't2':tech2,'t3':tech3})
sns.boxplot(data=techs)

df_techs = pd.melt(techs, value_vars=['t1','t2','t3'])
df_techs.columns = ['Technician','Measurement']
df_techs.head(5)
df_techs.to_csv("anova_test.csv",index=False)

formula = 'Measurement ~ Technician'

model = ols(formula, df_techs).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table)


############# two way anova

data = pd.read_csv("ToothGrowth.csv")
data= data.iloc[:,1:]
data.columns =['effect','supplier','dose']
data.head(5)

data[data.supplier=='VC'].effect.mean()
data[data.supplier=='OJ'].effect.mean()
data[data.dose==0.5].effect.mean()
data[data.dose==1.0].effect.mean()
data[data.dose==2.0].effect.mean()

sns.factorplot(data=data, x='dose', y='effect',hue='supplier',kind='box', legend=True)

sns.factorplot(data=data, x='dose', y='effect',kind='box', legend=True)
sns.factorplot(data=data, x='supplier', y='effect',kind='box', legend=True)


fig = interaction_plot(data.dose, data.supplier, data.effect,colors=['red','blue'], markers=['D','^'], ms=10)
# here we can see that OJ brand is more effective even in lower dose, and both brands are equally effective at higher dose
formula = 'effect ~ supplier + dose + supplier:dose'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2)
print (aov_table)

########### regression

'''
There are 14 attributes in each case of the dataset. They are:

    CRIM - per capita crime rate by town
    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per $10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - % lower status of the population
    MEDV - Median value of owner-occupied homes in $1000's
    
'''




housing = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
			header=None, sep='\s+')
housing.columns = ['CRIM','ZN','INDUS',
			'CHAS','NOX','RM',
			'AGE','DIS','RAD',
			'TAX','PTRATIO','B',
            'LSTAT','MEDV']

housing.shape
housing.head(5)

#housing.to_csv("housing_data.csv",index=False)
housing.isnull().sum()

x = housing.loc[:,'CRIM':'LSTAT']
y = housing.MEDV
features = x.columns



fig = plt.figure()
for i in range(0,len(features)):
    ax = plt.subplot(4, 4, i+1 )
    #sns.distplot(x.loc[:,features[i]],label=features[i])
    sns.regplot(x=x.loc[:,features[i]], y=y)


sns.regplot(x=x.loc[:,'AGE'], y=y)
sns.regplot(x=x.loc[:,'RM'], y=y)
sns.regplot(x=x.loc[:,'LSTAT'], y=y)

housing.MEDV.hist()

corr_matrix = housing.corr()
corr_matrix.shape
sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)

## here we check f values for each feature separately in a SIMPLE linear regression model
## FTEST is SS_Regression/SS_Error
for feat in features:
    result = sm.OLS( y, x.loc[:,feat] ).fit()
    print (feat, result.rsquared)

## multiple linear regression
## 1. check if regression is significant- done by FTEST using SS_Regression/SS_Error
## 2. check if individual feature is significant -- done by t-test using idea as drug effect before and after
### and compare individual coefficient to population of zero that means no change
result = sm.OLS( y, x ).fit()
result.summary()



##add features to check best r square at random addition of features
## this shows improvement in r square as we add features but at one point adding more feature wont help much
rsq_vals=[]
for i in range(1,len(features)):
    result= sm.OLS(y,x.iloc[:,0:i]).fit()
    print (i, result.rsquared)
    rsq_vals.append(result.rsquared)
plt.plot(rsq_vals)


### f test for model selection sequentially based on ftest ranking

f_test, _ = f_regression(x, y)
f_test /= np.max(f_test)

#for x,y in zip(x.columns,f_test): print(x,y)
rsqr =[]
current_features =[]
for i in range(0,len(f_test)):
    max_feat = np.where(f_test == f_test.max())[0][0]
    f_test[max_feat]=0.0
    temp = features[max_feat]
    current_features.append(temp)

    result = sm.OLS(y, x.loc[:, current_features]).fit()
    print (current_features, result.rsquared)
    rsqr.append(result.rsquared)
plt.plot(rsqr)

### thus here we can see that lstat, rm, and ptratio is already enough for max rsquare

## ftest on two model comparison
##https://en.wikipedia.org/wiki/F-test#F-test_of_the_equality_of_two_variances

feature_list1 =['LSTAT', 'RM', 'PTRATIO']
feature_list2 = ['LSTAT', 'RM', 'PTRATIO', 'INDUS']
model1  = sm.OLS(y, x.loc[:, feature_list1]).fit()
yhat1=model1.predict(x.loc[:, feature_list1])
model1_residuals = np.sum((y-yhat1)**2)


model2 = sm.OLS(y, x.loc[:, feature_list2]).fit()
yhat2=model2.predict(x.loc[:, feature_list2])
model2_residuals = np.sum((y-yhat2)**2)

n= len(y)
p1 =len(feature_list1)
p2 = len(feature_list2)
F = ((model1_residuals - model2_residuals)/(p2-p1))/(model2_residuals/(n-p2-1))
fcrit = f.pdf(0.05,p2-p1,n-p2)
print (F,fcrit)
# SINCE F is not greater than fcritical value, null is not rejected