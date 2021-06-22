def generate_sequence(id, seq_len, seq_cols):
    df_zeros=pd.DataFrame(np.zeros((seq_len-1,id.shape[1])),columns=id.columns)
    #print("df zeros",df_zeros)
    id=df_zeros.append(id,ignore_index=True)
    #print("id = ",id)
    data_array = id[seq_cols].values
    #print("data array",data_array)
    num_elements = data_array.shape[0]
    #print("num elemets",num_elements)
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_len), range(seq_len, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)
  
def gen_final_label(id, seq_len, seq_cols,final_label):
    df_zeros=pd.DataFrame(np.zeros((seq_len-1,id.shape[1])),columns=id.columns)
    id=df_zeros.append(id,ignore_index=True)
    data_array = id[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_len), range(seq_len, num_elements)):
        y_label.append(id[final_label][stop])
    return np.array(y_label)
  
  
  feature_cols = ['voltmean_24h','rotatemean_24h','pressuremean_24h','vibrationmean_24h',
                'voltsd_24h','rotatesd_24h',
                'pressuresd_24h','vibrationsd_24h','voltmean_5d','rotatemean_5d',
                'pressuremean_5d','vibrationmean_5d','voltsd_5d','rotatesd_5d',
                'pressuresd_5d','vibrationsd_5d','age','DI']

Target_cols = ['fail_label']

Final_train = Train_Data_copy[['machineID','voltmean_24h','rotatemean_24h','pressuremean_24h','vibrationmean_24h','voltsd_24h','rotatesd_24h','pressuresd_24h','vibrationsd_24h','voltmean_5d','rotatemean_5d',
                'pressuremean_5d','vibrationmean_5d','voltsd_5d','rotatesd_5d','pressuresd_5d','vibrationsd_5d','age','DI','fail_label']]

#Y = Train_Data_copy[Target_cols]

seq_len=50
seq_cols=feature_cols
from sklearn.model_selection import train_test_split
x_data= np.concatenate(list(list(generate_sequence(Train_Data_copy[Final_train['machineID']==id], seq_len, seq_cols)) for id in Final_train['machineID'].unique()))
x_data.shape
y_data = np.concatenate(list(list(gen_final_label(Final_train[Final_train['machineID']==id], 50, seq_cols,'fail_label')) for id in Final_train['machineID'].unique()))
y_data.shape
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,shuffle=True)
print("Training set size = ",x_train.shape, y_train.shape)
print("Testing set size = ",x_test.shape,y_test.shape)




  
  
