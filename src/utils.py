import sys
import numpy as np
import pandas as pd
import os
from src.exception import CustomException
import dill

#save_object saves the trained preprocessing pipeline into a file using dill.dump so it can be reused later for prediction without retraining.
def save_object(file_path,obj):
    try:
        #Extracts directory path.
        dir_path=os.path.dirname(file_path)

        #Creates directory if it doesn't exist
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)