import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, sys
import seaborn as sns

proximity_trace = pd.read_csv("data/proximityedgestimestamps.csv",sep=";",names=['Timestamp','Dev-1','Dev-2'],skipfooter=1,engine='python')
trace_list = list(zip(list(proximity_trace['Timestamp']),list(proximity_trace['Dev-1']),list(proximity_trace['Dev-2'])))
community_map = pd.read_csv('data/modularityclass.csv',sep=';',index_col=0,names=['Community'],engine='python')

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
devices['Community'] = community_map
###########################################

############ Gini Coeffiecient ############
def gini(dist):
    sort_dist = np.sort(dist)
    sum_of_absolute_differences = 0
    subsum = 0
    for i, x in enumerate(sort_dist):
        sum_of_absolute_differences += i * x - subsum
        subsum += x
    return np.divide(np.divide(sum_of_absolute_differences,subsum),len(sort_dist))
###########################################

########## Broadcast Algorithm 3 ##########
def broadcast_3(devices,source_id,x,y):
    devices_copy = devices.copy()
    devices_dict = devices_copy.drop(['# Devices Seen','Devices Seen'],axis=1).T.to_dict('list')
    devices_dict[source_id][0] = True
    device_chunk_count = 1
    time_90 = np.inf
    for idx, trace in enumerate(trace_list):
        dev1 = devices_dict[trace[1]]
        dev2 = devices_dict[trace[2]]
        if dev1[0] and not dev2[0]:
            prob = y if dev1[2] == dev2[2] else x
            if np.random.randint(1,101,size=1) < prob:
                devices_dict[trace[2]][0] = True
                devices_dict[trace[1]][1] = dev1[1]+1
                device_chunk_count = device_chunk_count + 1
        elif dev2[0] and not dev1[0]:
            prob = y if dev2[2] == dev1[2] else x
            if np.random.randint(1,101,size=1) < prob:
                devices_dict[trace[1]][0] = True
                devices_dict[trace[2]][1] = dev2[1]+1
                device_chunk_count = device_chunk_count + 1
        if device_chunk_count/len(devices_dict) > 0.9 and time_90 == np.inf:
            time_90 = trace[0]
    devices_copy2 = pd.DataFrame.from_dict(devices_dict,orient='index')
    devices_copy2.columns = ['Chunk Received','# Broadcasts','Community']
    gini_load = gini(devices_copy2['# Broadcasts'])
    return gini_load, time_90,(device_chunk_count-1),device_chunk_count*100/devices_copy.shape[0]
###########################################

def broadcast_3_analysis():
	######## Broadcast Algorithm 3 - Analysis 3.a ########
	source_id = 26
	x_list = np.array([1,2,5,10,20,30,40,45,50,55,60,70,80,90,95,98,99])
	y_list = np.array([1,2,5,10,20,30,40,45,50,55,60,70,80,90,95,98,99])
	analysis_df_3a = pd.DataFrame(columns=['T_Prob_Diff','T_Prob_Same','Gini_Load','Time_90','Chunk Copies','% Devices Reached'])

	### Uncomment only to recalculate entire analysis
	# for idx, x in enumerate(x_list):
	#     for idy, y in enumerate(y_list):
	#         gini_l, t, chnk, dev_reach = broadcast_3(devices,source_id,x,y)
	#         analysis_df_3a.loc[idx*len(x_list)+idy] = [x,y,gini_l,t,chnk,dev_reach]
	#         print(list(analysis_df_3a.loc[idx*len(x_list)+idy]))

	# analysis_df_3a.to_csv('out/Analysis_3a.csv',index=False)

	analysis_df_3a = pd.read_csv('out/Analysis_3a.csv')
	######################################################

	######### Broadcast Algorithm 3 - Analysis 3.a - Plot 1 ########
	analysis_df_3a_pivot1 = analysis_df_3a.pivot('T_Prob_Same','T_Prob_Diff','Gini_Load')
	analysis_df_3a_pivot2 = analysis_df_3a.pivot('T_Prob_Same','T_Prob_Diff','Time_90')
	inf_mask = np.isinf(analysis_df_3a_pivot2)
	fig, (ax1, ax2) = plt.subplots(2,1,figsize=(9,14),sharey='all')
	sns.heatmap(analysis_df_3a_pivot1,cmap="YlGnBu",linewidths=.3,ax=ax1,cbar_kws={"orientation": "vertical","label":"Gini Coeff. for Load on all nodes"})
	ax1.set_xlabel("Probability of Transmission to Different Community")
	ax1.set_ylabel("Probability of Transmission within Same Community")
	sns.heatmap(analysis_df_3a_pivot2,vmin=np.min(np.min(analysis_df_3a_pivot2[~inf_mask])),vmax=np.max(np.max(analysis_df_3a_pivot2[~inf_mask])),mask=inf_mask,cmap="YlGnBu",linewidths=.3,ax=ax2,cbar_kws={"orientation": "vertical","label":"Time for Broadcast to reach 90% Devices"})
	ax2.set_ylabel("Probability of Transmission within Same Community")
	ax2.set_xlabel("Probability of Transmission to Different Community ")
	plt.show()
	################################################################

	######## Broadcast Algorithm 3 - Analysis 3.b ########
	### Uncomment only to change the source device IDs (This needs the analysis to be redone)
	# source_ids_idx = random.sample(range(devices.shape[0]),100)
	# source_ids_idx_df = pd.DataFrame(data=source_ids_idx,columns=['id'])
	# source_ids_idx_df.to_csv('data/id_3.csv',index=False)

	source_ids_idx = list(pd.read_csv('data/id_3.csv')['id'])
	source_ids = list(devices.iloc[source_ids_idx].index)
	x_list = np.array([1,2,5,10,20,30,40,45,50,55,60,70,80,90,95,98,99])
	analysis_df_3b = pd.DataFrame(columns=['T_Prob_Diff','T_Prob_Same','Mean_Gini_Load',"Above 90% Reach",'Time_90_Mean','Time_90_Std','Chunk Copies','% Devices Reached'])

	### Uncomment only to recalculate entire analysis
	# for idx, x in enumerate(x_list):
	#     gini_l, time_90, chunk_copies, dev_reached = [], [], [], []
	#     for idd, source_id in enumerate(source_ids):
	#         g, t, chnk, dev = broadcast_3(devices,source_id,x,y)
	#         if(t != np.inf): time_90.append(t)
	#         gini_l.append(g)
	#         chunk_copies.append(chnk)
	#         dev_reached.append(dev)
	#     analysis_df_3b.loc[idx] = [x,100-x,np.nanmean(gini_l),len(time_90),np.mean(time_90),np.std(time_90),np.mean(chnk),np.mean(dev_reached)]
	#     print(list(analysis_df_3b.loc[idx]))

	# analysis_df_3b.to_csv('out/Analysis_3b.csv',index=False)

	analysis_df_3b = pd.read_csv('out/Analysis_3b.csv')
	######################################################

	######### Broadcast Algorithm 3 - Analysis 3.b - Plot 2 ########
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	plt.grid()
	x = np.arange(analysis_df_3b['T_Prob_Diff'].shape[0])
	r1 = ax1.bar(x,analysis_df_3b['Time_90_Mean'],width=0.4,color='#F15854',yerr=analysis_df_3b['Time_90_Std'],error_kw=dict(lw=1.5,capsize=1.5,capthick=1))
	r2 = ax2.bar(x+0.4,analysis_df_3b['Mean_Gini_Load'],width=0.4,color='#5DA5DA')
	ax1.set_xticks(x+0.25)
	ticklabels = [(a,b) for a,b in zip(list(analysis_df_3b['T_Prob_Diff']),list(analysis_df_3b['T_Prob_Same']))]
	ax1.set_xticklabels(ticklabels,fontsize=8)
	ax1.set_xlabel('Probability of Transmission to (Different Community, Same Community)')
	ax1.set_ylabel('Time')
	ax2.set_ylabel('Gini Coeff. for Load on Nodes')
	ax1.legend((r1[0],r2[0]),('Mean Time for Broadcast to reach 90% Devices','Mean Gini Coeff. for Load'),bbox_to_anchor=(0,1),loc=3,ncol=2)
	plt.show()
	################################################################



if __name__ == '__main__':
    if len(sys.argv) < 3:
        broadcast_3_analysis()
    else:
        print(broadcast_3(devices,float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])))