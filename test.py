import os
root_path = 'Total_Data_Simulation'
total_file_link = []
for i in os.listdir(root_path):
    total_file_link.append(root_path + f'/{i}')


print(total_file_link)