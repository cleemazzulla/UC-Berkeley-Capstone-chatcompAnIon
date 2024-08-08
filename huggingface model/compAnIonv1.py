# compAnIonv1.py>
# Setup environment for Spark
import os
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
#os.environ["SPARK_HOME"] = '/home/ubuntu/spark-3.5.1-bin-hadoop3'
#pip install transformers==4.31.0 -q
# importing pandas as pd
import pandas as pd

# Install spark-nlp
#!pip install spark-nlp
#import sparknlp
#from sparknlp.base import *
#from sparknlp.annotator import *
#from sparknlp.common import *
#from pyspark.sql.functions import *
#from pyspark.sql.functions import lit
#from pyspark.sql.window import Window
#from pyspark.sql.types import *
#from pyspark.ml.classification import LogisticRegression
#from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.mllib.evaluation import MulticlassMetrics
#from pyspark.ml import Pipeline
#from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer, OneHotEncoder, StringIndexer
#from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel, TrainValidationSplit, TrainValidationSplitModel
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#from pyspark.ml.linalg import Vectors, VectorUDT
#import pyspark.pandas as ps

# Import Tensorflow and BERT models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer
from transformers import TFBertModel

MAX_SEQUENCE_LENGTH = 400

def create_bert_classification_model(bert_model,
                                     num_train_layers=0,
                                     max_sequence_length=MAX_SEQUENCE_LENGTH,
                                     num_filters = [100, 100, 50, 25],
                                     kernel_sizes = [3, 4, 5, 10],
                                     hidden_size = 200,
                                     hidden2_size = 100,
                                     dropout = 0.1,
                                     learning_rate = 0.001,
                                     label_smoothing = 0.03
                                    ):
    """
    Build a simple classification model with BERT. Use the Pooler Output or CLS for classification purposes
    """
    if num_train_layers == 0:
        # Freeze all layers of pre-trained BERT model
        bert_model.trainable = False

    elif num_train_layers == 12:
        # Train all layers of the BERT model
        bert_model.trainable = True

    else:
        # Restrict training to the num_train_layers outer transformer layers
        retrain_layers = []

        for retrain_layer_number in range(num_train_layers):

            layer_code = '_' + str(11 - retrain_layer_number)
            retrain_layers.append(layer_code)


        #print('retrain layers: ', retrain_layers)

        for w in bert_model.weights:
            if not any([x in w.name for x in retrain_layers]):
                #print('freezing: ', w)
                w._trainable = False

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='input_ids')
    token_type_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='token_type_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='attention_mask')
                                    
    bert_inputs = {'input_ids': input_ids,
                   'token_type_ids': token_type_ids,
                   'attention_mask': attention_mask}

    bert_out = bert_model(bert_inputs)

    pooler_token = bert_out[1]
    cls_token = bert_out[0][:, 0, :]
    bert_out_avg = tf.math.reduce_mean(bert_out[0], axis=1)
    cnn_token = bert_out[0]

    conv_layers_for_all_kernel_sizes = []
    for kernel_size, filters in zip(kernel_sizes, num_filters):
        conv_layer = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(cnn_token)
        conv_layer = tf.keras.layers.GlobalMaxPooling1D()(conv_layer)
        conv_layers_for_all_kernel_sizes.append(conv_layer)

    conv_output = tf.keras.layers.concatenate(conv_layers_for_all_kernel_sizes, axis=1)

    # classification layer
    hidden = tf.keras.layers.Dense(hidden_size, activation='relu', name='hidden_layer')(conv_output)
    hidden = tf.keras.layers.Dropout(dropout)(hidden)

    hidden = tf.keras.layers.Dense(hidden2_size, activation='relu', name='hidden_layer2')(hidden)
    hidden = tf.keras.layers.Dropout(dropout)(hidden)
    
    classification = tf.keras.layers.Dense(1, activation='sigmoid',name='classification_layer')(hidden)

    classification_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[classification])

    classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                 # LOSS FUNCTION
                                 loss=tf.keras.losses.BinaryFocalCrossentropy(
                                   gamma=2.0, from_logits=False, apply_class_balancing=True, label_smoothing=label_smoothing
                                 ),
                                 # METRIC FUNCTIONS
                                 metrics=['accuracy']
                                 )
    return classification_model


f_one_or_zero = lambda x: 1 if x > 0.5 else 0

def run_inference_model(conversations):
        # Tokenize conversations with BERT tokenizer
        tokenized_input = tokenizer(conversations,
                                    max_length=MAX_SEQUENCE_LENGTH,
                                    truncation=True,
                                    padding='max_length',
                                    return_tensors='tf')
        bert_inputs = [tokenized_input.input_ids,
                       tokenized_input.token_type_ids,
                       tokenized_input.attention_mask]

        # Apply Model Prediction to testData
        y_pred = inference_model.predict(bert_inputs)
        #prediction = f_one_or_zero(y_pred)
        return y_pred



model_checkpoint = "bert-base-uncased"
 # Step 1: Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
# Step 2: Load Pretrained BERT model
bert_model = TFBertModel.from_pretrained(model_checkpoint)
# Stage 3: Create custom BERT model on top of the pretrained model
inference_model = create_bert_classification_model(bert_model=bert_model)
# Stage 4: Load Inference model with saved weights
save_path = 'bert_cnn_ensemble_resample_uncased_mdl.h5'
inference_model.load_weights(save_path)




