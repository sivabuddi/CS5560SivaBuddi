import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles
from IPython.display import display
import pyspark.sql.functions as fn


sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.csv("data.csv", header=True, inferSchema= True)
df.printSchema()
df.show(5)


#table_name = "Customers"
#df.write.format("parquet").saveAsTable(table_name)
#display(df)
#df.show(10)
table1= df.select("customerId",
                                "customerID",
                                "TechSupport",
                                "Contract",
                                "PaymentMethod",
                                "TotalCharges")
#display(table1)
#table1.show(10)
#showing top 10
table1.sort("Contract",ascending=True).show(10)
table2= df.select("customerId",
                                "Partner",
                                "Dependents",
                                "Contract",
                                "TotalCharges")
#display(table2)
#table2.show(10)
#showing top 10
table2.sort("Contract",ascending=True).show(10)
# Aggregate Functions
print("---------------Aggregate functions  filter, groupby and mean---------------------------------")
df.cache() \
    .filter("tenure in (1, 2, 5)") \
    .groupby("tenure") \
    .agg(fn.mean("TotalCharges").alias("Avg_Charges"), fn.mean("MonthlyCharges").alias("avg_monthly_charges")) \
    .show()


print("---------------Aggregate functions   Count---------------------------------")print("---------------Aggregate functions   Count---------------------------------")
df.groupby('tenure').agg(fn.count('Partner').alias('partner_count')).orderBy('tenure',ascending=True).show()

#print("---------------Joining Data Frames---------------------------------")

#table3 = table1.join(table2, table1.customerId == table2.customerId)
#table3.show(10)

#joinedDF = df.join(df, df.customerID == df.TotalCharges)
#joinedDF.show(10)

#print("------Column Correlation----------------")
#df.agg(fn.corr("tenure", "TotalCharges").alias('Corr_tenure_Part')).collect()

#show()
#df.groupby('borough').agg(f.count('borough').alias('count')).show()


#df.groupby('tenure').agg(fn.sum('TotalCharges').alias('sum_charges')).orderBy('TotalCharges', ascending=False).show()

#DT3.groupBy("package").count().sort("count", ascending = False).show(10)
# Filtering and Aggregation
#specific_columns_df.groupBy("package").count().sort("count", ascending = False).show(10)
#df.groupBy("PhoneService").count().sort("TotalCharges", ascending = False).show(10)
#df.write.save('/home/sivakumar/CS5560SivaBuddi/ICP4/test', format='parquet')

#df1 = sqlContext.read.load("/home/sivakumar/CS5560SivaBuddi/ICP4/test/*.parquet")
#df1.show(10)

#= spark.read.load("/home/sivakumar/CS5560SivaBuddi/ICP4")
#display(df)