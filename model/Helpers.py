import os
import pickle
import pathlib

def payload_data_generator(dir_path):

    out_list = os.listdir(dir_path)
    for file_nm in out_list:
        file_path = pathlib.PurePath(dir_path, file_nm)
        print(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        yield data

def load_model_output(payload):
    file_path = f'data/mip_outputs/{payload.dataset}/{payload.uniq_id}.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def model_inp_data_generator(dir_path):

    out_list = os.listdir(dir_path)
    for file_nm in out_list:
        file_path = pathlib.PurePath(dir_path, file_nm)
        print(file_path)
        with open(file_path, 'rb') as f:
            payload = pickle.load(f)
            
        mip_solver = load_model_output(payload)
        
        yield payload, mip_solver