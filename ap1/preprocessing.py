import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class Check:

    def __init__(self,arg):
        self.data =  arg
        self.data.remove(self.data[1])

    def col_len(self):

        if len(self.data) != 8:
            return False
        else:
            return True

    def transformation(self,data):
        data[1] = 1 if data[1] == 'male' else 0

        data[-1] =  data[-1] == 'S' and 2 or   ( data[-1] == 'C' and 0 or 1)
        return data

    def scaler(self,data):

        data = pd.DataFrame(data).astype(float)
        mm_scaler = MinMaxScaler()
        data = mm_scaler.fit_transform( data)

        return data

    def Run(self):

        if self.col_len():

            self.data = self.transformation(self.data)

            return pd.DataFrame(self.scaler(self.data).ravel())

        else:
            return "Malumotni notog'ri kirintdingiz"





