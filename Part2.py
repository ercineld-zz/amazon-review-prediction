%pyspark
#EXPLORATORY STATISTICS on ALL PRODUCT REVIEWS------------
#USE DATAFRAME AND SPARKSQL
from pyspark import SparkConf, SparkContext, SQLContext
conf = SparkConf()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
reviews = sqlContext.read.json("s3a://wer-amazon-review/kcore_5.json").cache();
products = sqlContext.read.json("s3a://wer-amazon-review/metadata.json").cache();

reviews.printSchema() 
reviews.show(10)
reviews.describe("overall").show()


%pyspark
#Top Rated Products
allratings = sc.textFile("s3a://wer-amazon-review/item_dedup.csv");
allRatingSplited = allratings.map(lambda x : x.split(",")).map(lambda x :(x[0],x[1],float(x[2]),long(x[3]))).cache()
#create rating data frame  
from pyspark.sql.functions import desc
productDataFrame = sqlContext.read.json("s3a://wer-amazon-review/metadata.json").cache();
ratingsDataFrame =  allRatingSplited.toDF(["user","asin","rating","timestamp"]);


%pyspark
#toprated products is joined with product metadata
topRatedProducts= ratingsDataFrame.groupby("asin").count().sort(desc("count"))
topRatedProducts = topRatedProducts.join(productDataFrame, on="asin")
topRatedProducts.createOrReplaceTempView("topProduct");


%pyspark
#Distribution of Ratings by Year
import datetime;
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,col

def totimestamptoYear(timestamp):
    if timestamp:
      value = datetime.datetime.fromtimestamp(float(timestamp))
      return value.strftime('%Y');
    return "1970";

totimestamptoYear_udf = udf(totimestamptoYear, StringType())

#add year columns    
ratingsWithYearDF = ratingsDataFrame.withColumn("year", totimestamptoYear_udf(col("timestamp")))
ratingsWithYearDF.createOrReplaceTempView("ratingsWithYearDF")
ratingsWithYearDF.show(10);


%sql
SELECT year, count(1) ratingCount FROM ratingsWithYearDF GROUP BY year ORDER BY year


%pyspark
#Review Distribution by Year
reviews = reviews.withColumn("unixReviewTime", col("unixReviewTime").cast("integer"))

#add year to reviews
reviewsWithYearDF = reviews.withColumn("year", totimestamptoYear_udf(col("unixReviewTime")) )
reviewsWithYearDF.createOrReplaceTempView("reviewsWithYearDF")
reviewsWithYearDF.show(10)


%sql
SELECT year, count(1) reviewCount FROM reviewsWithYearDF GROUP BY year ORDER BY year


%sql
SELECT year, count(distinct reviewerID) reviewerCount FROM reviewsWithYearDF GROUP BY year ORDER BY year


%pyspark
import pyspark.sql.functions as func

topReviewerOnReview  = reviewsWithYearDF.groupby("year","reviewerID").agg(func.count(func.lit(1)).alias("reviewCount")).sort(desc("reviewCount")).limit(1000)

topUserOnRating = ratingsWithYearDF.groupby("year","user").agg(func.count(func.lit(1)).alias("ratingCount")).sort(desc("ratingCount")).limit(1000)

joined = topReviewerOnReview.join(topUserOnRating ,(topReviewerOnReview.year == topUserOnRating.year) & (topReviewerOnReview.reviewerID == topUserOnRating.user))
joined.createOrReplaceTempView("joined")
joined.show()



