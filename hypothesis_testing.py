import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
import seaborn as sns
import random


#fisher exact test

'''
        School1  School2
male      8        12
female    12        8
'''
gender = [[7,3],[3,7]]
oddsratio, pvalue = stats.fisher_exact(gender)
print pvalue


#chi square test
''' 	Republican 	Democrat 	Totals
M 	    215 	    143 	    358
F 	    19      	64 	        83
Totals 	234 	    207 	    441'''

house = [ [ 314, 44 ], [  33,334 ] ]
chi2, p, ddof, expected = stats.chi2_contingency( house )
print p

diceroll1 = [9,10,12,11,8,10]
diceroll2 = [6,0,14,20,11,9]
dices =[diceroll1,diceroll2]
chi2, p = stats.chisquare( diceroll2 )
print p
chi2, p, ddof, expected = stats.chi2_contingency(dices)
print p



baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]

paired_sample = stats.ttest_rel(baseline, follow_up)
print paired_sample

### chit test for one way table -- phenotype 9:3:3:1 ratio
obs = [284,21,21,55]
exp = [214.3,71.4,71.4,23.8]
phenotype = [obs,exp]
chi2, p, ddof, expected = stats.chi2_contingency(phenotype)
print p

### chi test for two way table ( test for independence)
### drinking problem
       # low, mid, high times
police =[71,154,398]
nopolice =[4992,2808,2737]
drinking = [police,nopolice]
chi2, p, ddof, expected = stats.chi2_contingency(phenotype)
print p
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
print np.mean(group1)

pop_mean = 176
print stats.ttest_1samp(group1,pop_mean)


## unpaired t test for independent mean comparison
female = [63.8, 56.4, 55.2, 58.5, 64.0, 51.6, 54.6, 71.0]
male = [75.5, 83.9, 75.7, 72.5, 56.2, 73.4, 67.7, 87.9]

gender = pd.DataFrame({'male':male, 'female':female})
sns.boxplot(data=gender)
two_sample = stats.ttest_ind(male, female)
print two_sample

# assuming unequal population variances
two_sample_diff_var = stats.ttest_ind(male, female, equal_var=False)
print two_sample_diff_var


### paired t test
## idea is take difference of condition1 and condition2, and use this difference value to
## to do 1 sample t test to check if with population mean assumption as zero
## pain score before and after surgery
before = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
after = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]
painscore = pd.DataFrame({'before':before, 'after':after})
sns.boxplot(data=painscore)
paired_sample = stats.ttest_rel(before, after)
print  paired_sample



########### one way anova - comparing three or more sample means

tech1 = [random.randint(25,50) for x in range(0,100)]
tech2 = [random.randint(45,70) for x in range(0,100)]
tech3 = [random.randint(25,100) for x in range(0,100)]

techs = pd.DataFrame({'t1':tech1, 't2':tech2,'t3':tech3})
sns.boxplot(data=techs)

df_techs = pd.melt(techs, value_vars=['t1','t2','t3'])
df_techs.columns = ['Technician','Measurement']
df_techs.head(5)

formula = 'Measurement ~ Technician'

model = ols(formula, df_techs).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

