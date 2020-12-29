import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import tensorflow as tf


df = pd.read_csv('drive/My Drive/train.csv')

# Change 'srch_ci', 'srch_co', 'srch_local_date' from date to date of year
df['srch_ci_days_of_year'] = pd.DatetimeIndex(df['srch_ci']).dayofyear
df['srch_co_days_of_year'] = pd.DatetimeIndex(df['srch_co']).dayofyear
df['srch_local_date_days_of_year'] = pd.DatetimeIndex(df['srch_local_date']).dayofyear

trainData, valData = model_selection.train_test_split(df, test_size = 0.3, stratify = y, random_state=42)
trainData2 = trainData.copy()
valData2 = valData.copy()

# Select numerical features
num = ['srch_visitor_visit_nbr', 'srch_los', 'srch_bw', 'srch_adults_cnt', 'srch_children_cnt',
       'prop_price_without_discount_usd', 'prop_room_capacity', 'prop_review_score', 'prop_review_count',
       'srch_ci_days_of_year', 'srch_local_date_days_of_year']

# scaling numerical feature
for num_col in num_cols:
    scaler = StandardScaler()
    scaler.fit(trainData2[[num_col]])
    trainData2[num_col] = scaler.transform(trainData2[num_cols])
    valData2[num_col] = scaler.transform(valData2[num_cols])


# modify 'prop_starrating' and 'srch_visitor_loc_region'
# so they can be inferred to string datatype by tf.data.make_csv_dataset later
trainData2['prop_starrating'] = list(map(lambda x : str(x) + 'star', trainData2['prop_starrating']))
valData2['prop_starrating'] = list(map(lambda x : str(x) + 'star', valData2['prop_starrating']))

trainData2['srch_visitor_loc_region'] = list(map(lambda x: 'region' + str(x), trainData2['srch_visitor_loc_region']))
valData2['srch_visitor_loc_region'] = list(map(lambda x: 'region' + str(x), valData2['srch_visitor_loc_region']))

trainData2.to_csv('trainData2.csv')
valData2.to_csv('valData2.csv')

# load training data and validation data as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='prop_booking_bool',
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset

train_dataset = get_dataset('trainData2.csv')
val_dataset = get_dataset('valData2.csv')

# check train_dataset and val_dataset is loaded
example_batch, example_label = next(iter(train_dataset))
print(example_label)
print(example_batch['srch_ci_day'])

# process numerical column
numerical_columns = [tf.feature_column.numeric_column('srch_visitor_visit_nbr'),
                     tf.feature_column.numeric_column('srch_los'),
                     tf.feature_column.numeric_column('srch_bw'),
                     tf.feature_column.numeric_column('srch_adults_cnt'),
                     tf.feature_column.numeric_column('srch_children_cnt'),
                     tf.feature_column.numeric_column('prop_price_without_discount_usd'),
                     tf.feature_column.numeric_column('prop_room_capacity'),
                     tf.feature_column.numeric_column('prop_review_score'),
                     tf.feature_column.numeric_column('prop_review_count'),
                     tf.feature_column.numeric_column('srch_ci_days_of_year'),
                     tf.feature_column.numeric_column('srch_local_date_days_of_year')]

cat_cols_low = ['srch_hcom_destination_id', 'srch_ci_day', 'srch_co_day', 'srch_mobile_bool', 'srch_device',
                'prop_travelad_bool', 'prop_dotd_bool', 'prop_imp_drr', 'prop_brand_bool',
                'prop_starrating', 'prop_super_region', 'prop_country', 'prop_hostel_bool']

cat_cols_high = ['srch_visitor_loc_country', 'srch_visitor_loc_region', 'srch_visitor_loc_city', 'srch_posa_country',
                 'prop_market_id', 'prop_submarket_id']

categorical_columns = []

# one hot encode low cardinality categorical feature
for cat_col_low in cat_cols_low:
    categorical_columns.append(
        tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key = cat_col_low, vocabulary_list = trainData2[cat_col_low].unique())))

# embedding high cardinality categorical features
for cat_col_high in cat_cols_high:
    categorical_columns.append(
        tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key = cat_col_high, vocabulary_list = trainData2[cat_col_high].unique()), 20))

# cross features between prop_country and srch_visitor_loc_country
prop_country_col = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'prop_country', vocabulary_list = trainData2['prop_country'].unique())

srch_visitor_loc_country_col = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'srch_visitor_loc_country', vocabulary_list = trainData2['srch_visitor_loc_country'].unique())

crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column(
    [prop_country_col, srch_visitor_loc_country_col], 20))

# define inputs for keras model
inputs = {
    'srch_visitor_visit_nbr': tf.keras.layers.Input(name='srch_visitor_visit_nbr', shape=(), dtype='float32'),
    'srch_los': tf.keras.layers.Input(name='srch_los', shape=(), dtype='float32'),
    'srch_bw': tf.keras.layers.Input(name='srch_bw', shape=(), dtype='float32'),
    'srch_adults_cnt': tf.keras.layers.Input(name='srch_adults_cnt', shape=(), dtype='float32'),
    'srch_children_cnt': tf.keras.layers.Input(name='srch_children_cnt', shape=(), dtype='float32'),
    'prop_price_without_discount_usd': tf.keras.layers.Input(name='prop_price_without_discount_usd', shape=(), dtype='float32'),
    'prop_room_capacity': tf.keras.layers.Input(name='prop_room_capacity', shape=(), dtype='float32'),
    'prop_review_score': tf.keras.layers.Input(name='prop_review_score', shape=(), dtype='float32'),
    'prop_review_count': tf.keras.layers.Input(name='prop_review_count', shape=(), dtype='float32'),
    'srch_ci_days_of_year': tf.keras.layers.Input(name='srch_ci_days_of_year', shape=(), dtype='float32'),
    'srch_local_date_days_of_year': tf.keras.layers.Input(name='srch_local_date_days_of_year', shape=(), dtype='float32'),

    'srch_hcom_destination_id': tf.keras.layers.Input(name='srch_hcom_destination_id', shape=(), dtype='int32'),
    'srch_ci_day': tf.keras.layers.Input(name='srch_ci_day', shape=(), dtype='int32'),
    'srch_co_day': tf.keras.layers.Input(name='srch_co_day', shape=(), dtype='int32'),
    'srch_mobile_bool': tf.keras.layers.Input(name='srch_mobile_bool', shape=(), dtype='int32'),
    'srch_device': tf.keras.layers.Input(name='srch_device', shape=(), dtype='string'),
    'prop_travelad_bool': tf.keras.layers.Input(name='prop_travelad_bool', shape=(), dtype='int32'),
    'prop_dotd_bool': tf.keras.layers.Input(name='prop_dotd_bool', shape=(), dtype='int32'),
    'prop_imp_drr': tf.keras.layers.Input(name='prop_imp_drr', shape=(), dtype='int32'),
    'prop_brand_bool': tf.keras.layers.Input(name='prop_brand_bool', shape=(), dtype='int32'),
    'prop_starrating': tf.keras.layers.Input(name='prop_starrating', shape=(), dtype='string'),
    'prop_super_region': tf.keras.layers.Input(name='prop_super_region', shape=(), dtype='string'),
    'prop_country': tf.keras.layers.Input(name='prop_country', shape=(), dtype='string'),
    'prop_hostel_bool': tf.keras.layers.Input(name='prop_hostel_bool', shape=(), dtype='int32'),

    'srch_visitor_loc_country': tf.keras.layers.Input(name='srch_visitor_loc_country' , shape=(), dtype='string'),
    'srch_visitor_loc_region': tf.keras.layers.Input(name='srch_visitor_loc_region' , shape=(), dtype='string'),
    'srch_visitor_loc_city': tf.keras.layers.Input(name='srch_visitor_loc_city' , shape=(), dtype='string'),
    'srch_posa_country': tf.keras.layers.Input(name='srch_posa_country' , shape=(), dtype='string'),
    'prop_market_id': tf.keras.layers.Input(name='prop_market_id' , shape=(), dtype='int32'),
    'prop_submarket_id': tf.keras.layers.Input(name='prop_submarket_id' , shape=(), dtype='int32'),
}

# wide and deep model architecture
# deep part for all input features
deep = tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns)(inputs)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dropout(0.2)(deep)
deep = tf.keras.layers.Dense(128, activation='relu')(deep)
deep = tf.keras.layers.Dropout(0.2)(deep)

# wide part for cross features
wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)
both = tf.keras.layers.concatenate([deep, wide])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
model = tf.keras.Model(inputs, output_layer)

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[tf.keras.metrics.Precision(thresholds=0.1),
             tf.keras.metrics.Recall(thresholds=0.1),
             tf.keras.metrics.AUC(curve='ROC')]
)

# train the model
model.fit(train_dataset, epochs=3)

# evaluate the model
val_loss, val_precision, val_recall, val_roc_auc = model.evaluate(val_dataset)
print('\n\nVal Loss {}, Val Precision {}, Val Recall {}, Val ROC AUC {}'.format(val_loss, val_precision,
                                                                                val_recall, val_roc_auc))


