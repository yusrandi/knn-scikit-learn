from testing import KaeNeNTesting
from naive_bayes import NaiveBayesTesting
import pandas as pd 

# text = "kode otp sampai sekarang tdak masuk, gmna saya mau login"
# resKnnPredict = KaeNeNTesting().ngetest(text= text)
# resNBPredict = NaiveBayesTesting().ngetest(text= text)

# print("resKnnPredict ", resKnnPredict)
# print("resNBPredict ", resNBPredict)


dataReviews = pd.read_csv('all_apm3.csv')
# filtered_df = dataReviews[(dataReviews['score'].astype(float) < 4) & (dataReviews['impact'] == 'Negative')]
# print(len(filtered_df))

results = []
for index, row in dataReviews.iterrows():
    result = []
    # if (int(row['score']) < 4) & (row['impact'] == 'Negative'):
    print(index)
    text = row['content']
    print(text)
    
    resPredict = KaeNeNTesting().ngetest(text= text)
    resNBPredict = NaiveBayesTesting().ngetest(text= text)
    print(f"resPredict {resPredict}")
    print(f"resNBPredict {resNBPredict}")

    
    result.append(text)
    result.append(resPredict)
    result.append(resNBPredict)
    results.append(result)

df_results = pd.DataFrame(results)
headers = ['Text', 'KNN', 'NB']
df_results.to_excel("result.xlsx", index=False, header=headers)
