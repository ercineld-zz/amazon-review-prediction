from pyspark import SparkConf, SparkContext
import collections
conf = SparkConf().setMaster("local").setAppName("AmazonRatings")
sc = SparkContext.getOrCreate()
lines=sc.textFile("file:/usr/local/Cellar/apache-spark/datasets/ratings_Electronics.csv")

values=lines.map(lambda line: line.split(','))
#number of records
reviews=lines.count()
#number of unique reviewers
user=values.map(lambda value: value[0])
users=user.distinct().count()
#number of unique products
product=values.map(lambda value: value[1])
products=product.distinct().count()
print("Total number of Reviews: ", reviews)
print("Total number of Unique Reviewers: ", users)
print("Total number of Unique Products: ",products)
#overall average rating for Electronics category
from pyspark.mllib.stat import Statistics
ratings=values.map(lambda value: float(value[2]))
print(ratings.mean())
print(ratings.variance())

#distribution of ratings
import collections
import matplotlib.pyplot as plt
ratings=values.map(lambda value: float(value[2]))
ratingdist=ratings.countByValue()
sortedratingdist=collections.OrderedDict(sorted(ratingdist.items()))
for key, value in sortedratingdist.items():
	print(key,value)
#sum check the calculation of rating distribution
sum(sortedratingdist.values())
#plot distribution
x,y=zip(*sortedratingdist.items())
plt.title("Distibution of Ratings for 7,842,482 Reviews")
plt.bar(x,y)
plt.show()

#average ratings per product
averageByproduct=values.map(lambda values: (values[1], float(values[2]))).groupByKey().mapValues(lambda ratings: sum(ratings)/float(len(ratings)))
#average product ratings distribution
productratings=averageByproduct.map(lambda x: round(x[1],1))
productratingdist=productratings.countByValue()
sortedrating=collections.OrderedDict(sorted(productratingdist.items()))
for key, value in sortedrating.items():
	print(key,value)
#sum check
sum(sortedrating.values())
#plot distribution
x,y=zip(*sortedrating.items())
plt.title("Distribution of Overall Ratings for 476,002 Electronics Products")
plt.bar(x,y)
plt.show()

#number of ratings per products
productCounts = product.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
productCountsSorted = productCounts.sortBy(lambda a:a[1],ascending=False).collect()
#sum check
sum([pair[1] for pair in productCountsSorted])
result=productCountsSorted[:20]
for key, value in result:
	print(key, value)
result=productCountsSorted[:5]
#plot distribution
import numpy as np
plt.title("Most Rated Top 5 Products")
x,y=zip(*result)
l=np.arange(5)
plt.bar(l,y)
plt.xticks(l,x)
plt.show()
