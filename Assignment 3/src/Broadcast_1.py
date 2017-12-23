import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, sys
import seaborn as sns

proximity_trace = pd.read_csv("data/proximityedgestimestamps.csv",sep=";",names=['Timestamp','Dev-1','Dev-2'],skipfooter=1,engine='python')
trace_list = list(zip(list(proximity_trace['Timestamp']),list(proximity_trace['Dev-1']),list(proximity_trace['Dev-2'])))

############ Devices DataFrame ############
device_id = list(set(pd.unique(proximity_trace['Dev-1'])).union(set(pd.unique(proximity_trace['Dev-2']))))
devices = pd.DataFrame(data=[(False,0,0,set())]*len(device_id),index=device_id,columns=['Chunk Received','# Broadcasts','# Devices Seen','Devices Seen'])
for idx in range(proximity_trace.shape[0]):
    dev1_idx = proximity_trace.loc[idx, 'Dev-1']
    dev2_idx = proximity_trace.loc[idx, 'Dev-2']
    t1 = set(devices.loc[dev1_idx, 'Devices Seen'])
    t1.add(dev2_idx)
    devices.set_value(dev1_idx,'Devices Seen',t1)
    t2 = set(devices.loc[dev2_idx, 'Devices Seen'])
    t2.add(dev1_idx)
    devices.set_value(dev2_idx,'Devices Seen',t2)
devices['# Devices Seen'] = devices.apply(lambda row: len(row['Devices Seen']),axis=1)
###########################################

########## Broadcast Algorithm 1 ##########
def broadcast_1(devices,source_id,max_broadcasts):
    devices_dict = devices.drop(['# Devices Seen','Devices Seen'],axis=1).T.to_dict('list')
    devices_dict[source_id][0] = True
    device_chunk_count = 1
    time_90 = np.inf
    for idx, trace in enumerate(trace_list):
        dev1 = devices_dict[trace[1]]
        dev2 = devices_dict[trace[2]]
        if dev1[0] and dev1[1] < max_broadcasts and not dev2[0]:
            devices_dict[trace[2]][0] = True
            devices_dict[trace[1]][1] = dev1[1]+1
            device_chunk_count = device_chunk_count + 1
        elif dev2[0] and dev2[1] < max_broadcasts and not dev1[0]:
            devices_dict[trace[1]][0] = True
            devices_dict[trace[2]][1] = dev2[1]+1
            device_chunk_count = device_chunk_count + 1
        if device_chunk_count/len(devices_dict) > 0.9 and time_90 == np.inf:
            time_90 = trace[0]
    return time_90,(device_chunk_count-1)
###########################################

def broadcast_1_analysis():
    ####### Broadcast Algorithm 1 - Analysis 1.a,b #######
    source_id = 26
    analysis_df_1ab = pd.DataFrame(columns=['K','Time_90','Chunk Copies','% Devices Reached'])

    ### Uncomment only to recalculate entire analysis
    # for idx in range(100):
    #     time_90, chunk_copies = broadcast_1(devices,source_id,(idx+1))
    #     analysis_df_1ab.loc[idx] = [(idx+1),time_90,chunk_copies,(chunk_copies+1)*100/devices.shape[0]]
    #     print(list(analysis_df_1ab.loc[idx]))

    # analysis_df_1ab.to_csv('out/Analysis_1ab.csv',index=False)

    analysis_df_1ab = pd.read_csv('out/Analysis_1ab.csv')
    ######################################################

    ####### Broadcast Algorithm 1 - Analysis 1.a,b - Plot 1 #######
    sns.reset_defaults()
    fig, ax = plt.subplots()
    plt.plot(analysis_df_1ab['K'],analysis_df_1ab['Time_90'])
    cutoff_idx = analysis_df_1ab.shape[0] - np.argmin((np.diff(analysis_df_1ab.loc[analysis_df_1ab['Time_90'] < np.inf,'Time_90']) > -1000)[::-1])
    cutoff_pt = (cutoff_idx,int(analysis_df_1ab.loc[cutoff_idx,'Time_90']))
    rx = plt.plot(cutoff_pt[0],cutoff_pt[1],'rx')
    ax.annotate('({0}, {1})'.format(cutoff_pt[0],cutoff_pt[1]),xy=cutoff_pt,textcoords='data')
    plt.ylabel('Time for Broadcast to reach 90% Devices')
    plt.xlabel('Max Broadcasts per Device (K)')
    plt.legend((rx),('No noticeable change after this point',))
    plt.grid()
    plt.show()
    ################################################################

    ####### Broadcast Algorithm 1 - Analysis 1.a,b - Plot 2 #######
    sns.set()
    plt.plot(analysis_df_1ab['K'],analysis_df_1ab['Chunk Copies'])
    plt.ylabel('# Chunk Copies')
    plt.xlabel('Max Broadcasts per Device (K)')
    plt.show()
    ################################################################

    ######## Broadcast Algorithm 1 - Analysis 1.c ########
    ### Uncomment only to change the source device IDs (This needs the analysis to be redone)
    # source_ids_idx = random.sample(range(devices.shape[0]),100)
    # source_ids_idx_df = pd.DataFrame(data=source_ids_idx,columns=['id'])
    # source_ids_idx_df.to_csv('data/id_1.csv',index=False)

    source_ids_idx = list(pd.read_csv('data/id_1.csv')['id'])
    source_ids = list(devices.iloc[source_ids_idx].index)
    analysis_df_1c = pd.DataFrame(columns=['K','# Above 90% Reach','Time_90_Mean','Time_90_std','Chunk Copies Mean','Chunk Copies Std','% Devices Reached Mean','% Devices Reached Std'])
    analysis_df_1c_dump = pd.DataFrame(columns=['K','Source Device ID','Time_90','Chunk Copies','% Devices Reached'])

    ### Uncomment only to recalculate entire analysis
    # for idk, k in enumerate(range(40,71)):
    #     time_90, chunk_copies, devices_reached = [], [], []
    #     for idx, source_id in enumerate(source_ids):
    #         time_90_i, chunk_copies_i = broadcast_1(devices,source_id,k)
    #         analysis_df_1c_dump.loc[(idk*100+idx)] = [k,source_id,time_90_i,chunk_copies_i,(chunk_copies_i+1)*100/devices.shape[0]]
    #         if (time_90_i != np.inf): time_90.append(time_90_i)
    #         chunk_copies.append(chunk_copies_i)
    #         devices_reached.append((chunk_copies_i+1)*100/devices.shape[0])
    #         print(list(analysis_df_1c_dump.loc[(idk*100+idx)]))
    #     analysis_df_1c.loc[idk] = [k,len(time_90),np.mean(time_90),np.std(time_90),np.mean(chunk_copies),np.std(chunk_copies),np.mean(devices_reached),np.std(devices_reached)]
    #     print("#######\n",list(analysis_df_1c.loc[idk]),"\n#######")

    # analysis_df_1c.to_csv('out/Analysis_1c.csv',index=False)
    # analysis_df_1c_dump.to_csv('out/Analysis_1c_dump.csv',index=False)

    analysis_df_1c = pd.read_csv('out/Analysis_1c.csv')
    analysis_df_1c_dump = pd.read_csv('out/Analysis_1c_dump.csv')
    ######################################################

    ######### Broadcast Algorithm 1 - Analysis 1.c - Plot 1 ########
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.grid()
    r1 = ax1.bar(analysis_df_1c['K'],analysis_df_1c['Time_90_Mean'],width=0.4,color='#60BD68',yerr=analysis_df_1c['Time_90_std'])
    r2 = ax2.bar(analysis_df_1c['K']+0.4,analysis_df_1c['# Above 90% Reach'],width=0.4,color='#5DA5DA')
    ax1.set_ylabel('Mean Time')
    ax2.set_ylabel('# Devices')
    ax1.set_xlabel('Max Broadcasts per Device (K)')
    ax1.legend((r1[0],r2[0]),('Mean Time for Broadcast to reach 90% Devices','# Source Devices that could reach 90% Devices'),bbox_to_anchor=(0,1),loc=3)
    plt.show()
    ################################################################

if __name__ == '__main__':
    if len(sys.argv) < 3:
        broadcast_1_analysis()
    else:
        print(broadcast_1(devices,float(sys.argv[1]),float(sys.argv[2])))