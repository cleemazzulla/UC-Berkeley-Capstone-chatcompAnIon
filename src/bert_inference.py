# Setup environment for Spark
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = '/home/ubuntu/spark-3.5.1-bin-hadoop3'

# importing pandas as pd
import pandas as pd

# Install spark-nlp
#!pip install spark-nlp
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.sql.functions import *
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer, OneHotEncoder, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel, TrainValidationSplit, TrainValidationSplitModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.pandas as ps

# Start Spark Session with Spark NLP
spark = sparknlp.start(gpu=True)
#spark = sparknlp.start(gpu=False)


# Step 1: Transforms raw texts to `document` annotation
document = DocumentAssembler()\
              .setInputCol("merged_text")\
              .setOutputCol("document")\
              .setCleanupMode("shrink")

# Step 2: Encodes text into high dimensional vectors
bert_sent = BertSentenceEmbeddings.pretrained('sent_small_bert_L8_512')\
              .setInputCols(["document"])\
              .setOutputCol("sentence_embeddings")

# Stage 3: Load saved BERT Classifier Model into Spark Classifier
TrainedClassifierDLModel = ClassifierDLModel.load('BertSentenceClfModel')

# Stage 4: Generate prediction Pipeline with loaded Model
ld_pipeline = Pipeline(stages=[document, bert_sent, TrainedClassifierDLModel])
ld_pipeline_model = ld_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))



def run_inference_model(conversations):

    data = [[conversations]]
    # Create the spark and pandas DataFrame
    pd_df = pd.DataFrame(data, columns=['merged_text'])
    spark_df = spark.createDataFrame(pd_df)

    # Apply Model Transform to testData
    preds_test = ld_pipeline_model.transform(spark_df)

    preds_test_df = preds_test.select("class.result").toPandas()
    prediction = preds_test_df['result'].apply(lambda x : int(x[0]))

    return prediction[0]


conversations = " Hey baby, sorry I’m late. Lol I’m not your baby. To me you are :) We just met a couple days ago. You have to earn it. Challenge accepted."
pd0 = run_inference_model(conversations)
print('Prediction0: ', pd0)

conversations = "We just met a couple days ago. You have to earn it. Challenge accepted."
pd1 = run_inference_model(conversations)
print('Prediction1: ', pd1)



