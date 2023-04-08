import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureTransformerBase:
    final_columns = list() # if given, use exact list of cols with exact order
    label_col_name = None
    def __init__(self, df):
        self.df = df
        assert self.label_col_name not in self.final_columns
        if self.label_col_name is None:
            raise NotImplementedError
        self.df_new = self.transform_dataframe(self.df.copy(), train=True)
        if len(self.final_columns) == 0:
            self.final_columns = list(self.df_new.drop(columns=self.label_col_name, 
                                            errors='ignore').columns)

        
    def transform_dataframe(self, df, train=True):
        raise NotImplementedError
        
    def get_X(self):
        return self.df_new[self.final_columns].to_numpy(dtype=np.float64)

    def get_y(self):
        return self.df_new[self.label_col_name].to_numpy(dtype=np.float64)        


class FeatureTransformerLocationNaive(FeatureTransformerBase):
    final_columns = [
        'year', 'mth', 'num_rooms', 'is_executive', 'is_multi_gen', 
        'storey_level', 'lease_commence_date', 'floor_area_sqm'
    ]
    label_col_name = 'resale_price'
    
    def transform_dataframe(self, df, train=True):
        # month
        df[['year', 'mth']] = df['month'].str.split('-', expand=True)
        df = df.astype({'year': 'int', 'mth': 'int'})
        # flat_type
        def parse_flat_type(s_type):
            if not isinstance(s_type, str):
                return pd.Series((None, None, None))
            if s_type[0] in ['1', '2', '3', '4', '5']:
                return pd.Series((int(s_type[0]), False, False))
            elif s_type == 'executive':
                return pd.Series((0, True, False))
            elif s_type == 'multi generation':
                return pd.Series((0, False, True))
            else:
                return pd.Series((None, None, None))
        df[['num_rooms', 'is_executive', 'is_multi_gen']] = df['flat_type'].apply(parse_flat_type)
        # storey_range
        df['storey_level'] = df['storey_range'].apply(lambda s: int(s[0:2]))
        # flat_model
        if train: # training set, remove invalid records
            index_to_drop = df[df['flat_model'] == '2 room'].index
            df.drop(index=index_to_drop, inplace=True)
        return df


class FeatureTransformerZZY(FeatureTransformerBase):
    label_col_name = 'resale_price'

    def __init__(self, df):
        ## encoder and scaler to be set during training and re-used for prediction.
        self.encoder = None
        self.scaler = None
        super().__init__(df)
        
    
    def transform_dataframe(self, df, train=True):
        df['year'] = df['month'].str.split('-').apply(lambda x:x[0]).astype('int')
        df['month'] = df['month'].str.split('-').apply(lambda x:x[1]).astype('int')
        df['flat_type'] = df['flat_type'].str.replace('-', ' ').astype('category')
        df['storey_range_avg'] = df['storey_range'].str.split(' to ').apply(
            lambda x:(int(x[1])+int(x[0]))/2
        )
        df['is_low_floor'] = df['storey_range_avg'].apply(lambda x: 1 if x < 6 else 0)
        # convert string to categorical variables
        df['town'] = df['town'].astype('category')
        df['block'] = df['block'].astype('category')
        df['street_name'] = df['street_name'].astype('category')
        df['flat_model'] = df['flat_model'].astype('category')
        df['subzone'] = df['subzone'].astype('category')
        df['planning_area'] = df['planning_area'].astype('category')
        df['region'] = df['region'].astype('category')
        # TODO: remaining columns: 'block', 'street_name' ,'subzone'. Maybe can use auxiliary data
        x_num_cols = ['month', 'year', 'storey_range_avg', 'is_low_floor', 'floor_area_sqm', 'lease_commence_date', 'latitude', 'longitude', 'elevation']
        x_cat_cols = ['flat_type', 'town', 'flat_model', 'planning_area', 'region']
        ## When training, need to store encoder and scaler to object
        if train:
            self.encoder = OneHotEncoder(drop='first').fit(df[x_cat_cols])
        ## When predicting, use the encoder and scaler obtained during training to transform
        if self.encoder is None:
            raise ValueError("During predicting, encoder not found!")

        X_cat_encoded = pd.DataFrame(
            self.encoder.transform(df[x_cat_cols]).toarray(), 
            columns=self.encoder.get_feature_names_out()
        )
        X_encoded = pd.concat([df[x_num_cols], X_cat_encoded], axis=1)
        if train:
            self.scaler = StandardScaler().fit(X_encoded)
        if self.scaler is None:
            raise ValueError("During predicting, scaler not found!")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X_encoded), columns=X_encoded.columns
        )
        return pd.concat((X_scaled, df[self.label_col_name]), axis=1)