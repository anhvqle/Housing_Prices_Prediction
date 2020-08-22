import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn import linear_model
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")

    #demonstrateHelpers(trainDF)
    #correlation(trainDF)
    #print(trainDF.describe())
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    #doExperiment1(trainInput, trainOutput, predictors)
    #doExperiment2(trainInput, trainOutput, predictors,.18)
    #doExperiment3(trainInput, trainOutput, predictors, .14)
    #doExperiment4(trainInput, trainOutput, predictors , .05)
    #tuneLinearRidge(trainInput, trainOutput, predictors)
    tuneGBR(trainInput, trainOutput, predictors)
    #tuneLasso(trainInput, trainOutput, predictors)
    
    #makeJointGraph(trainInput, trainOutput, predictors)
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment1(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    #print("CV Average Score:", cvMeanScore)
    return cvMeanScore
def doExperiment2(trainInput, trainOutput, predictors, x):
    alg = linear_model.Ridge(alpha=x, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    #print("CV Average Score:", cvMeanScore)
    #print(cvMeanScore)
    return cvMeanScore
def doExperiment3(trainInput, trainOutput, predictors, x):
    gbrt=GradientBoostingRegressor(n_estimators=100,learning_rate=x) 
    cvMeanScore = model_selection.cross_val_score(gbrt, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    #print("CV Average Score:", cvMeanScore)
    #print(cvMeanScore)
    return cvMeanScore
def doExperiment4(trainInput, trainOutput, predictors , x):
    alg = Lasso(alpha =x, random_state=1)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    #print("CV Average Score:", cvMeanScore)
    return cvMeanScore
    
# ===============================================================================
'''
VISUALIZATION and TUNING! 
Runs the algorithm on the testing set and writes the results to a csv file.
https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/
many ideas came from this website 
'''
def tuneGBR(trainInput,trainOutput,predictors):
    
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment3(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.plot(alphaList, acc)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Tunning of GBR')
    
def tuneLinearRidge(trainInput,trainOutput,predictors):
    
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment2(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.plot(alphaList, acc)
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.title('Tunning of Linear Ridge')
    
def tuneLasso(trainInput,trainOutput,predictors):
    
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.1,2,.1)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment4(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.plot(alphaList, acc)
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.title('Tunning of Lasso')
    
def makeJointGraph(trainInput,trainOutput,predictors):
    f, axes = plt.subplots(1,3,figsize=(10,10))
    (ax_1, ax_2,ax_3) =axes
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment3(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    ax_1.xlabel = 'Learning Rate'
    ax_1.ylabel ='Accuracy'
    ax_1.set_title('Accuracy of Gradient Boosting Regression')
    ax_1.plot(alphaList,acc)
    
    accLR = alphaList.map(lambda x: doExperiment2(trainInput, trainOutput, predictors,x) )
   
    ax_2.xlabel = 'Alpha'
    ax_2.ylabel ='Accuracy'
    ax_2.set_title('Accuracy of Linear Ridge')
    ax_2.plot(alphaList,accLR)
    tuneSeq2 = np.arange(.001,.01,.01)
    alphaList2 = pd.Series(tuneSeq2,index=tuneSeq2)
    accLasso = alphaList2.map(lambda x: doExperiment4(trainInput, trainOutput, predictors,x) )
    ax_3.xlabel = 'alpha'
    ax_3.ylabel ='Accuracy'
    ax_3.set_title('Accuracy of Lasso')
    ax_3.plot(alphaList,accLasso)

'''
using https://seaborn.pydata.org/generated/seaborn.heatmap.html 
'''
def correlation(DF):
    numeric=DF.select_dtypes(exclude='object')
    correlation=numeric.corr()
    numericCorr=correlation['SalePrice'].sort_values(ascending=False).head(10).to_frame()
    print(numericCorr)
    sns.heatmap(correlation, annot=False, fmt=".2f",cmap ='PuBuGn')
    plt.show()
    
    

def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    #alg = LinearRegression()
    #alg = Lasso(alpha =0.001, random_state=1)
    alg = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1) 
    #alg = linear_model.Ridge(alpha=0.14, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    
    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    '''
    Dropping values: 
        
    Dropping missing values of attribute w/ 90% of the data missing
    ''' 
    
   
    trainDF = trainDF.dropna(thresh=len(trainDF)*0.9, axis=1)
    testDF = testDF.dropna(thresh=len(testDF)*0.9, axis=1)
    missingCat, missingNumeric = getAttrsWithMissingValues(testDF)
    #print(missingCat)
    #print(missingNumeric)
    
    #taking care of Numeric missing values
    for i in missingNumeric:
        if(i=='GarageYrBlt'):
            fillNaWithMean(trainDF,testDF,'GarageYrBlt')
        trainDF.loc[:,i] = trainDF.loc[:,i].fillna(0)
        testDF.loc[:,i] = testDF.loc[:,i].fillna(0)
    #filling with 0 wouldnt make sense
    #taking care of Catagorical missing values
    #print(testDF[missingCat].isnull().sum()) #gave us an idea of how to fill values 
    for i in missingCat:
        if(testDF[i].isnull().sum()<10):
            fillNaWithMode(trainDF,testDF,i)
        else:
            trainDF.loc[:,i] = trainDF.loc[:,i].fillna('None')
            testDF.loc[:,i] = testDF.loc[:,i].fillna('None')
            
            
    squareFootage = ['LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea','LowQualFinSF','1stFlrSF', '2ndFlrSF','TotalSQF']
    rooms = ['TotRmsAbvGrd','KitchenAbvGr','BedroomAbvGr','TotalBath']
    ageCharacteristics =['YearBuilt','age'] #will ad years old and years since last remodeled
    qualitative= ['ExterQual','ExterCond','OverallQual','OverallCond','BsmtQual','BsmtCond','KitchenQual','HeatingQC','Remodeled']
    location = ['Neighborhood','Condition1','Condition2']
    one=[] 
  
    '''
    Feature Engineering:
    after discussion we decided changing remodeled to T/F would be better than the year
    Creating 'totalBathrooms' to encapsulate all the bath rooms and remove attributes
    creating total area to encompass how large the house is 
    creating yearsold to provide better idea of age
    '''
    
    #creating a totalBaths
    trainDF['TotalBath'] = trainDF.apply(lambda row: row['FullBath']+row['HalfBath']*(.5)+row['BsmtHalfBath']*(.5)+row['BsmtFullBath'],axis=1)
    testDF['TotalBath'] = testDF.apply(lambda row: row['FullBath']+row['HalfBath']*(.5)+row['BsmtHalfBath']*(.5)+row['BsmtFullBath'],axis=1)
    #creating total inside SQFootage
    trainDF['TotalSQF']= +trainDF['GarageArea']+ trainDF['TotalBsmtSF'] + trainDF['1stFlrSF'] + trainDF['2ndFlrSF'] + trainDF['GrLivArea'] 
    testDF['TotalSQF']= +testDF['GarageArea']+ testDF['TotalBsmtSF'] + testDF['1stFlrSF'] + testDF['2ndFlrSF'] + testDF['GrLivArea'] 
    #convert year of remodel to YEs or No remodeled
    yearRemodtoTF(trainDF,testDF)
    #creating year sold
    trainDF['age'] = 2019-trainDF['YearBuilt']
    testDF['age'] = 2019-trainDF['YearBuilt']
    
    
    # changes year remodeled to be just true=1 or false=0
    encodeNeighborHood(trainDF,testDF) # changes neighborhoods to be ordinal 
    encodeCondition1(trainDF,testDF)
    encodeCondition2(trainDF,testDF)#changes condition to be ordinal 
    #one = one+ oneHot(trainDF,testDF,'HouseStyle',one) #onehot encodes housestyle 
    #one = one+ oneHot(trainDF,testDF,'RoofMatl', one) #NEW 
    one = one+ oneHot(trainDF,testDF,'Foundation',one) #NEW
    one = one+ binaryEncode(trainDF, testDF, 'RoofStyle',one)
    predictors =  qualitative + rooms+location+one+squareFootage+ageCharacteristics+one
    
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    for i in one:
        trainInput.loc[:,i] = trainInput.loc[:,i].fillna(0)
        testInput.loc[:,i] = testInput.loc[:,i].fillna(0)
    
    #normalize(trainInput,'LotArea')
    #normalize(testInput,'LotArea')
    '''
    Changing catagorical data to  ordinal numeric values
    important for linear regression 
    '''
    l1 = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual']
    for i in l1:
        ordinal_To_Numeric(trainInput,testInput,i)
    
    '''
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
def standardize(trainDF,testDF, cols):
     trainDF.loc[:,cols] = trainDF.loc[:,cols].apply(lambda x: (x-trainDF.loc[:,cols].mean())/(trainDF.loc[:,cols].std()), axis=1)
     testDF.loc[:,cols] = testDF.loc[:,cols].apply(lambda x: (x-trainDF.loc[:,cols].mean())/(trainDF.loc[:,cols].std()), axis=1)
     
def fillNaWithMode(trainDF,testDF,col):
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    testDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
def fillNaWithMean(trainDF,testDF,col):
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(round(trainDF.loc[:, col].mean()))
    testDF.loc[:, col] = trainDF.loc[:, col].fillna(round(trainDF.loc[:, col].mean()))
    
def encodeNeighborHood(trainDF,testDF):#converting categorical to ordinal
    #this function was created to convert neighborhood category into ordinal based on the 
    #relative price of houses in that area, the groupby function was found in pandas documentation
    neighbors = trainDF.loc[:,['Neighborhood','SalePrice']]
    neighbrs2 = neighbors.groupby(['Neighborhood']).agg(['mean','count'])
    #print(neighbrs2)
    #print(neighbrs2['SalePrice','mean'].sort_values())
    neighDict = {'Neighborhood':{'NoRidge':24,'NridgHt':23,'StoneBr':22,'Timber':21,'Veenker':20,'Somerst':19,'ClearCr':18,
                                   'Crawfor':17,'CollgCr':16,'Blmngtn':15,'Gilbert':14,'NWAmes':13,'SawyerW':12,'Mitchel':11,'NAmes':10,
                                   'NPkVill':9,'SWISU':8,'Blueste':7,'Sawyer':6,'OldTown':5,'Edwards':4,'BrkSide':3,'BrDale':2,'IDOTRR':1,
                                   'MeadowV':0}}
    trainDF.replace(neighDict, inplace=True)
    testDF.replace(neighDict, inplace=True)
def encodeCondition2(trainDF,testDF):
    #code used to examine conditions 
    #condDF = trainDF.loc[:,['Condition1','SalePrice']]
    #conDF2 = condDF.groupby(['Condition1']).agg(['mean','count'])
    #print(conDF2)
    #print(conDF2['SalePrice','mean'].sort_values())
    conditionDict = {'Condition2':{'PosA':8,'PosN':7,'RRNn':6,'RRNe':5,'Norm':4,'RRAn':3,'Feedr':2,'RRAe':1,'Artery':0}}
    trainDF.replace(conditionDict,inplace=True)
    testDF.replace(conditionDict,inplace=True)    
def encodeCondition1(trainDF,testDF):
    #code used to examine conditions 
    #condDF = trainDF.loc[:,['Condition1','SalePrice']]
    #conDF2 = condDF.groupby(['Condition1']).agg(['mean','count'])
    #print(conDF2)
    #print(conDF2['SalePrice','mean'].sort_values())
    conditionDict = {'Condition1':{'PosA':8,'PosN':7,'RRNn':6,'RRNe':5,'Norm':4,'RRAn':3,'Feedr':2,'RRAe':1,'Artery':0}}
    trainDF.replace(conditionDict,inplace=True)
    testDF.replace(conditionDict,inplace=True)
def binaryEncode(trainDF, testDF, col,binaryCol):
    
    lb_style = LabelBinarizer()
    lb_results = lb_style.fit_transform(trainDF[col])
    binarizedDF = pd.DataFrame(lb_results, columns=lb_style.classes_)

    trainDF= trainDF.drop(col,axis = 1)
    trainDF = trainDF.join(binarizedDF)
    style = LabelBinarizer()
    result = style.fit_transform(testDF[col])
    binarizedTestDF = pd.DataFrame(result, columns=style.classes_)
    #print(binarizedTestDF)
    testDF= testDF.drop(col,axis = 1)
    testDF = testDF.join(binarizedTestDF)
    binaryCol = binaryCol + list(binarizedTestDF.columns)
    return binaryCol
def yearRemodtoTF(trainDF,testDF):
    trainDF['YearRemodAdd'] = trainDF.apply(lambda row: 0 if row['YearRemodAdd']==row['YearBuilt'] else 1,axis=1)
    testDF['YearRemodAdd'] = testDF.apply(lambda row: 0 if row['YearRemodAdd']==row['YearBuilt'] else 1,axis=1)
    trainDF.rename(columns = {'YearRemodAdd':'Remodeled'}, inplace = True)
    testDF.rename(columns = {'YearRemodAdd':'Remodeled'}, inplace = True)
    #yearsOld = 2019-trainDF.loc[:,'YearBuilt']
    #trainDF['YearBuilt']= yearsOld
    #yearsOldTest = 2019-testDF.loc[:,'YearBuilt']
    #testDF['YearBuilt'] = yearsOldTest
    

# =============================================================================
def oneHot(trainDF,testDF, col,one): # creates dummies for categorical data
    
    one_hot = pd.get_dummies(trainDF[col])
    trainDF= trainDF.drop(col,axis = 1)
    trainDF = trainDF.join(one_hot)

    one_hotT = pd.get_dummies(testDF[col])
    testDF= testDF.drop(col,axis = 1)
    testDF = testDF.join(one_hotT)
    one = one + list(one_hot.columns)
    
    #updates predictors with columns 
    return one
def ordinal_To_Numeric(trainDF,testDF,col):#transforms nominal rankings into ascending ordinal values		
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4 if v=="Ex" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3 if v=="Gd" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2 if v=="TA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1 if v=="Fa" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=="Po" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: trainDF.loc[:, col].mode().loc[0] if v=="NA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: trainDF.loc[:, col].mode().loc[0] if v=="None" else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
# ===============================================================================  
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    
    valueCatCountSeries = df.select_dtypes(include='object').count(axis=0)  # 0 to count down the rows
    numCatCases = df.select_dtypes(include='object').shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCatCases - valueCatCountSeries)  # A Series showing the number of missing values, for each attribute
    catWithMissingValues = missingSeries[missingSeries != 0].index
    #improvement to seperate numeric and catagorical data 
    valueNumCountSeries = df.select_dtypes(exclude='object').count(axis=0)
    numNumCases = df.select_dtypes(exclude='object').shape[0]
    missingNumSeries = (numNumCases - valueNumCountSeries)
    numWithMissingValues = missingNumSeries[missingNumSeries != 0].index
    return catWithMissingValues, numWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()
