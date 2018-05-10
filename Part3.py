%pyspark

from pyspark import SparkConf, SparkContext,SQLContext

#conf.set('spark.executor.instances', 5)
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

productsJsonText = sc.textFile("s3a://wer-amazon-review/metadata.json")
reviews = sqlContext.read.json("s3a://wer-amazon-review/kcore_5.json")


%pyspark
reviewsCountByBooks = reviews.groupby('asin').count().rdd.collectAsMap();


%pyspark
import json,ast;

def toProductDictinary(productJsonText):
  productJson =  eval(productJsonText);
  
  productDict = {'asin':productJson['asin']}
  
  if "categories" in productJson:
    productDict['category'] = productJson['categories'][0][0];
    if len(productJson['categories'][0]) > 1:
      print("xx");    
  else:
    productDict['category']="unknown";
  return (productJson["asin"],productDict);
productsDictList = productsJsonText.map(lambda x : toProductDictinary(x));


%pyspark
def addReviewCountToProductDictinary(asin ,productDictionary):
    productDictionary['reviewCount'] = 0;
    if asin in  reviewsCountByBooks:
      productDictionary['reviewCount'] = reviewsCountByBooks[asin];
       
    return (asin,productDictionary);

productsDictList = productsDictList.map(lambda x :addReviewCountToProductDictinary(x[0],x[1]));
print(productsDictList.take(10))


%pyspark
productReviewCountDict= sc.broadcast(reviewsCountByBooks);


%pyspark
from pyspark.sql.functions import length,udf,col;
from pyspark.sql.types import StringType,FloatType,LongType 

def getProductCategory(asin):
    return "unknown"   
    
def getProductReviewCounty(asin):
    if asin in  productReviewCountDict.value:
      return int(productReviewCountDict.value[asin]);
    return 0;     

getProductCategory_udf = udf(getProductCategory, StringType())
getProductReviewCounty_udf = udf(getProductReviewCounty, StringType())

reviews = reviews.withColumn('reviewLength', length('reviewText'))
reviews = reviews.withColumn('summaryLength', length('summary'))
reviews = reviews.withColumn('category' ,getProductCategory_udf(col('asin')))
reviews = reviews.withColumn('reviewCount' ,getProductReviewCounty_udf(col('asin')))
reviews = reviews.withColumn("reviewCount", col("reviewCount").cast("int"))
reviews.show()


%pyspark
reviews = reviews.withColumn("total_votes",reviews["helpful"].getItem(1)).withColumn("helpful", reviews["helpful"].getItem(0))
reviews.show()


%pyspark
reviews.createOrReplaceTempView("reviews2")
reviewdf = sqlContext.sql("SELECT CASE WHEN helpful/total_votes> 0.6 THEN 1 ELSE 0 END AS Helpful_Score, reviewText, reviewLength, summaryLength,category,reviewCount, overall FROM reviews2")
reviewdf.show(5)


%pyspark
import pyspark.sql.functions as func
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StopWordsRemover

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
wordsData = tokenizer.transform(reviewdf)

#StopWords Exclution
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
wordsData = remover.transform(wordsData)
wordsData = wordsData.select('filtered', 'summaryLength', 'reviewLength','Helpful_Score','overall','reviewCount')
wordsData.show(5)


%pyspark
#Hash TF on tokenized data
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=50)
featurizedData = hashingTF.transform(wordsData)

#TF-IDF Vectorizer
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

data = rescaledData.select("features","summaryLength", "reviewLength","reviewCount", "overall","Helpful_Score")
data.show(5)
type(data)

data.createOrReplaceTempView("data2")
data = sqlContext.sql("SELECT features as tfidfvector, summaryLength, reviewLength,reviewCount, overall, Helpful_Score as label FROM data2")
data.show(5)
type(data)


%pyspark
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#Combining features as one features column
assembler = VectorAssembler(inputCols=["tfidfvector","summaryLength","reviewLength","reviewCount", "overall"], outputCol="features")
output = assembler.transform(data)
finaldf = output.select("features", "label")
finaldf.show(5)


%pyspark

#Split the data into training and test sets (30% held out for testing)
train, test = finaldf.randomSplit([0.7, 0.3], seed=12345)
train.show(5)
test.show(5)


%pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(finaldf)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(finaldf)

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=20)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
                               
# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)

print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]

print(rfModel)  # summary only


%pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(finaldf)

featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(finaldf)
    
#Split the data into training and test sets (30% held out for testing)
#train, test = finaldf.randomSplit([0.7, 0.3], seed=12345)

# Train a GBT model.
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=20)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

# Train model.  This also runs the indexers.
model = pipeline.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  # summary only


%pyspark
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Split the data into training and test sets (30% held out for testing)
#train, test = finaldf.randomSplit([0.7, 0.3], seed=12345)

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))


%pyspark
sc.stop()
