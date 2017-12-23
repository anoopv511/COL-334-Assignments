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

################ CCDF Plot ################
degree_distribution = np.bincount(devices['# Devices Seen'])
s = degree_distribution.sum()
ccdf = 1 - (degree_distribution.cumsum(0)/s)
plt.plot(range(len(ccdf)),ccdf,'b-',markersize=3)
plt.xscale('log')
plt.yscale('log')
plt.ylim([0,1])
plt.ylabel('CCDF')
plt.xlabel('Degree')
plt.show()
###########################################

############## Device Classes #############
def get_device_classes(ccdf,s,l,devices):
    devices_copy = devices.copy()
    super_node_lower = np.argmax(ccdf <= s/100)
    bottom_node_upper = np.argmin(ccdf >= 1-(l/100))-1
    devices_copy.set_value(devices_copy['# Devices Seen'] >= super_node_lower,'Class','Super-Node')
    devices_copy.set_value(devices_copy['# Devices Seen'] <= bottom_node_upper,'Class','Weak-Node')
    return devices_copy
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

########## Broadcast Algorithm 2 ##########
def broadcast_2(devices,source_id,s,l,x,y):
    devices['Class'] = ['Ordinary']*devices.shape[0]
    devices_copy = get_device_classes(ccdf,s,l,devices)
    devices_dict = devices_copy.drop(['# Devices Seen','Devices Seen','Community'],axis=1).T.to_dict('list')
    devices_dict[source_id][0] = True
    device_chunk_count = 1
    time_90 = np.inf
    for idx, trace in enumerate(trace_list):
        dev1 = devices_dict[trace[1]]
        dev2 = devices_dict[trace[2]]
        if dev1[0] and not dev2[0]:
            prob = 100 if dev2[2] == 'Weak-Node' else x if dev2[2] == 'Super-Node' else y
            if np.random.randint(1,101,size=1) < prob:
                devices_dict[trace[2]][0] = True
                devices_dict[trace[1]][1] = dev1[1]+1
                device_chunk_count = device_chunk_count + 1
        elif dev2[0] and not dev1[0]:
            prob = 100 if dev1[2] == 'Weak-Node' else x if dev1[2] == 'Super-Node' else y
            if np.random.randint(1,101,size=1) < prob:
                devices_dict[trace[1]][0] = True
                devices_dict[trace[2]][1] = dev2[1]+1
                device_chunk_count = device_chunk_count + 1
        if device_chunk_count/len(devices_dict) > 0.9 and time_90 == np.inf:
            time_90 = trace[0]
    devices_copy2 = pd.DataFrame.from_dict(devices_dict,orient='index')
    devices_copy2.columns = ['Chunk Received','# Broadcasts','Class']
    gini_load_super = gini(devices_copy2[devices_copy2['Class'] == 'Super-Node']['# Broadcasts'])
    gini_load_ordinary = gini(devices_copy2[devices_copy2['Class'] == 'Ordinary']['# Broadcasts'])
    gini_load_weak = gini(devices_copy2[devices_copy2['Class'] == 'Weak-Node']['# Broadcasts'])
    mean_load_super = np.mean(devices_copy2[devices_copy2['Class'] == 'Super-Node']['# Broadcasts'])
    mean_load_ordinary = np.mean(devices_copy2[devices_copy2['Class'] == 'Ordinary']['# Broadcasts'])
    mean_load_weak = np.mean(devices_copy2[devices_copy2['Class'] == 'Weak-Node']['# Broadcasts'])
    return gini_load_super,gini_load_ordinary,gini_load_weak,mean_load_super,mean_load_ordinary,mean_load_weak,time_90,(device_chunk_count-1),device_chunk_count*100/devices_copy.shape[0]
###########################################

def broadcast_2_analysis():
    ######## Broadcast Algorithm 2 - Analysis 2.a ########
	source_id = 26
	s = 0.5
	l = 70
	T_Prob_Super = np.array([1,2,5,10,20,40,50,60,80,90,95,98,99])
	T_Prob_Ordinary = np.array([1,2,5,10,20,40,50,60,80,90,95,98,99])
	analysis_df_2a = pd.DataFrame(columns=['T_Prob_Super','T_Prob_Ordinary','Gini_Load_Super','Gini_Load_Ordinary','Gini_Load_Weak','Mean_Load_Super','Mean_Load_Ordinary','Mean_Load_Weak','Time_90','Chunk Copies','% Devices Reached'])

	### Uncomment only to recalculate entire analysis
	# for ids, tps in enumerate(T_Prob_Super):
	#     for ido, tpo in enumerate(T_Prob_Ordinary):
	#         analysis_df_2a.loc[ids*len(T_Prob_Super)+ido] = [tps,tpo] + list(broadcast_2(devices,source_id,s,l,tps,tpo))
	#         print(list(analysis_df_2a.loc[ids*len(T_Prob_Super)+ido]))

	# analysis_df_2a.to_csv('out/Analysis_2a.csv',index=False)

	analysis_df_2a = pd.read_csv('out/Analysis_2a.csv')
	######################################################

    ######### Broadcast Algorithm 2 - Analysis 2.a - Plot 1 ########
	analysis_df_2a_pivot1 = analysis_df_2a.pivot('T_Prob_Super','T_Prob_Ordinary','Mean_Load_Super')
	analysis_df_2a_pivot2 = analysis_df_2a.pivot('T_Prob_Super','T_Prob_Ordinary','Gini_Load_Super')
	analysis_df_2a_pivot3 = analysis_df_2a.pivot('T_Prob_Super','T_Prob_Ordinary','Mean_Load_Ordinary')
	analysis_df_2a_pivot4 = analysis_df_2a.pivot('T_Prob_Super','T_Prob_Ordinary','Gini_Load_Ordinary')
	fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(9,12),nrows=2,ncols=2)
	sns.heatmap(analysis_df_2a_pivot1,cmap="YlGnBu",linewidths=.3,ax=ax1,cbar_kws={"orientation": "horizontal","label":"Mean Load on Super Node"})
	ax1.set_xlabel("Probability of Transmission to Ordinary Node")
	ax1.set_ylabel("Probability of Transmission to Super Node")
	sns.heatmap(analysis_df_2a_pivot2,cmap="YlGnBu",linewidths=.3,ax=ax2,cbar_kws={"orientation": "horizontal","label":"Gini Coeff. for Load on Super Node"})
	ax2.set_xlabel("Probability of Transmission to Ordinary Node")
	ax2.set_ylabel("")
	sns.heatmap(analysis_df_2a_pivot3,cmap="YlGnBu",linewidths=.3,ax=ax3,cbar_kws={"orientation": "horizontal","label":"Mean Load on Ordinary Node"})
	ax3.set_xlabel("Probability of Transmission to Ordinary Node")
	ax3.set_ylabel("Probability of Transmission to Super Node")
	sns.heatmap(analysis_df_2a_pivot4,cmap="YlGnBu",linewidths=.3,ax=ax4,cbar_kws={"orientation": "horizontal","label":"Gini Coeff. for Load on Ordinary Node"})
	ax4.set_xlabel("Probability of Transmission to Ordinary Node")
	ax4.set_ylabel("")
	plt.show()
	################################################################

    ######### Broadcast Algorithm 2 - Analysis 2.a - Plot 2 ########
	analysis_df_2a_ign = analysis_df_2a[(analysis_df_2a["T_Prob_Ordinary"] + analysis_df_2a["T_Prob_Super"]) == 100]
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	plt.grid()
	x = np.arange(analysis_df_2a_ign['T_Prob_Super'].shape[0])
	r1 = ax1.bar(x,analysis_df_2a_ign['Time_90'],width=0.3,color='#F15854')
	r2 = ax2.bar(x+0.3,analysis_df_2a_ign['Mean_Load_Super'],width=0.2,color='#5DA5DA')
	r3 = ax2.bar(x+0.5,analysis_df_2a_ign['Mean_Load_Ordinary'],width=0.2,color='#B276B2')
	r4 = ax2.bar(x+0.7,analysis_df_2a_ign['Mean_Load_Weak'],width=0.2,color='#60BD68')
	ax1.set_xticks(x+0.3)
	ticklabels = [(a,b) for a,b in zip(list(analysis_df_2a_ign['T_Prob_Super']),list(analysis_df_2a_ign['T_Prob_Ordinary']))]
	ax1.set_xticklabels(ticklabels)
	ax1.set_xlabel('Probability of Transmission to (Super Node, Ordinary Node)')
	ax1.set_ylabel('Time')
	ax2.set_ylabel('Mean Load on Nodes')
	ax1.legend((r1[0],r2[0],r3[0],r4[0]),('Time for Broadcast to reach 90% Devices','Mean Load on Super Nodes','Mean Load on Ordinary Nodes','Mean Load on Weak Nodes'),bbox_to_anchor=(0,1),loc=3,ncol=2)
	plt.show()
	################################################################

	######### Broadcast Algorithm 2 - Analysis 2.a - Plot 3 ########
	analysis_df_2a_pivot5 = analysis_df_2a.pivot('T_Prob_Super','T_Prob_Ordinary','Time_90')
	inf_mask = np.isinf(analysis_df_2a_pivot5)
	fig,ax = plt.subplots()
	sns.heatmap(analysis_df_2a_pivot5,vmin=np.min(np.min(analysis_df_2a_pivot5[~inf_mask])),vmax=np.max(np.max(analysis_df_2a_pivot5[~inf_mask])),mask=inf_mask,cmap="YlGnBu",linewidths=.3,ax=ax,cbar_kws={"orientation": "vertical","label":"Time for Broadcast to reach 90% Devices"})
	ax.set_xlabel("Probability of Transmission to Ordinary Node")
	ax.set_ylabel("Probability of Transmission to Super Node")
	plt.show()
	################################################################

    ######## Broadcast Algorithm 2 - Analysis 2.b ########
	### Uncomment only to change the source device IDs (This needs the analysis to be redone)
	# source_ids_idx = random.sample(range(devices.shape[0]),100)
	# source_ids_idx_df = pd.DataFrame(data=source_ids_idx,columns=['id'])
	# source_ids_idx_df.to_csv('data/id_2.csv',index=False)

	source_ids_idx = list(pd.read_csv('data/id_2.csv')['id'])
	source_ids = list(devices.iloc[source_ids_idx].index)
	s = 0.5
	l = 70
	T_Prob_Super = np.array([1,2,5,10,20,40,50,60,80,90,95,98,99])
	T_Prob_Ordinary = np.array([1,2,5,10,20,40,50,60,80,90,95,98,99])
	analysis_df_2b = pd.DataFrame(columns=['T_Prob_Super','T_Prob_Ordinary','Mean_Gini_Load_Super','Mean_Gini_Load_Ordinary','Mean_Gini_Load_Weak','Mean_Mean_Load_Super','Mean_Mean_Load_Ordinary','Mean_Mean_Load_Weak','# Above 90% Reach','Time_90_Mean','Time_90_Std','Chunk Copies','% Devices Reached'])

	### Uncomment only to recalculate entire analysis
	# for ids, tps in enumerate(T_Prob_Super):
	#     for ido, tpo in enumerate(T_Prob_Ordinary):
	#         ginild_super, ginild_ordinary, ginild_weak, meanld_super, meanld_ordinary, meanld_weak, time_90, chunk_copies, dev_reached = [], [], [], [], [], [], [], [], []
	#         for idd, source_id in enumerate(source_ids):
	#             gini_s,gini_o,gini_w,mean_s,mean_o,mean_w,t,chnk,dev = broadcast_2(devices,source_id,s,l,tps,tpo)
	#             if(t != np.inf): time_90.append(t)
	#             ginild_super.append(gini_s)
	#             ginild_ordinary.append(gini_o)
	#             ginild_weak.append(gini_w)
	#             meanld_super.append(mean_s)
	#             meanld_ordinary.append(mean_o)
	#             meanld_weak.append(mean_w)
	#             chunk_copies.append(chnk)
	#             dev_reached.append(dev)
	#         analysis_df_2b.loc[ids*len(T_Prob_Super)+ido] = [tps,tpo,np.nanmean(ginild_super),np.nanmean(ginild_ordinary),np.nanmean(ginild_weak),np.nanmean(meanld_super),np.nanmean(meanld_ordinary),np.nanmean(meanld_weak),len(time_90),np.mean(time_90),np.std(time_90),np.mean(chunk_copies),np.mean(dev_reached)]
	#         print(list(analysis_df_2b.loc[ids*len(T_Prob_Super)+ido]))

	# analysis_df_2b.to_csv('out/Analysis_2b.csv',index=False)

	analysis_df_2b = pd.read_csv('out/Analysis_2b.csv')
	######################################################

    ######### Broadcast Algorithm 2 - Analysis 2.b - Plot 1 ########
	analysis_df_2b_pivot1 = analysis_df_2b.pivot('T_Prob_Super','T_Prob_Ordinary','Mean_Mean_Load_Super')
	analysis_df_2b_pivot2 = analysis_df_2b.pivot('T_Prob_Super','T_Prob_Ordinary','Mean_Gini_Load_Super')
	analysis_df_2b_pivot3 = analysis_df_2b.pivot('T_Prob_Super','T_Prob_Ordinary','Mean_Mean_Load_Ordinary')
	analysis_df_2b_pivot4 = analysis_df_2b.pivot('T_Prob_Super','T_Prob_Ordinary','Mean_Gini_Load_Ordinary')
	fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(9,12),nrows=2,ncols=2)
	sns.heatmap(analysis_df_2b_pivot1,cmap="YlGnBu",linewidths=.3,ax=ax1,cbar_kws={"orientation": "horizontal","label":"Mean of Mean Load on Super Node"})
	ax1.set_xlabel("Probability of Transmission to Ordinary Node")
	ax1.set_ylabel("Probability of Transmission to Super Node")
	sns.heatmap(analysis_df_2b_pivot2,cmap="YlGnBu",linewidths=.3,ax=ax2,cbar_kws={"orientation": "horizontal","label":"Mean of Gini Coeff. for Load on Super Node"})
	ax2.set_xlabel("Probability of Transmission to Ordinary Node")
	ax2.set_ylabel("")
	sns.heatmap(analysis_df_2b_pivot3,cmap="YlGnBu",linewidths=.3,ax=ax3,cbar_kws={"orientation": "horizontal","label":"Mean of Mean Load on Ordinary Node"})
	ax3.set_xlabel("Probability of Transmission to Ordinary Node")
	ax3.set_ylabel("Probability of Transmission to Super Node")
	sns.heatmap(analysis_df_2b_pivot4,cmap="YlGnBu",linewidths=.3,ax=ax4,cbar_kws={"orientation": "horizontal","label":"Mean of Gini Coeff. for Load on Ordinary Node"})
	ax4.set_xlabel("Probability of Transmission to Ordinary Node")
	ax4.set_ylabel("")
	plt.show()
	################################################################

    ######### Broadcast Algorithm 2 - Analysis 2.b - Plot 2 ########
	analysis_df_2b_ign = analysis_df_2b[(analysis_df_2b["T_Prob_Ordinary"] + analysis_df_2b["T_Prob_Super"]) == 100]
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	plt.grid()
	x = np.arange(analysis_df_2b_ign['T_Prob_Super'].shape[0])
	r1 = ax1.bar(x,analysis_df_2b_ign['Time_90_Mean'],width=0.2,color='#F15854')
	r2 = ax2.bar(x+0.2,analysis_df_2b_ign['Mean_Mean_Load_Super'],width=0.2,color='#5DA5DA')
	r3 = ax2.bar(x+0.4,analysis_df_2b_ign['Mean_Mean_Load_Ordinary'],width=0.2,color='#B276B2')
	r4 = ax2.bar(x+0.6,analysis_df_2b_ign['Mean_Mean_Load_Weak'],width=0.2,color='#60BD68')
	ax1.set_xticks(x+0.3)
	ticklabels = [(a,b) for a,b in zip(list(analysis_df_2b_ign['T_Prob_Super']),list(analysis_df_2b_ign['T_Prob_Ordinary']))]
	ax1.set_xticklabels(ticklabels)
	ax1.set_xlabel('Probability of Transmission to (Super Node, Ordinary Node)')
	ax1.set_ylabel('Time')
	ax2.set_ylabel('Mean Load on Nodes')
	ax1.legend((r1[0],r2[0],r3[0],r4[0]),('Time for Broadcast to reach 90% Devices','Mean Load on Super Nodes','Mean Load on Ordinary Nodes','Mean Load on Weak Nodes'),bbox_to_anchor=(0,1),loc=3,ncol=2)
	plt.show()
	################################################################

	######### Broadcast Algorithm 2 - Analysis 2.b - Plot 3 ########
	analysis_df_2b_pivot5 = analysis_df_2b.pivot('T_Prob_Super','T_Prob_Ordinary','Time_90_Mean')
	analysis_df_2b_pivot6 = analysis_df_2b.pivot('T_Prob_Super','T_Prob_Ordinary','# Above 90% Reach')
	inf_mask = np.isinf(analysis_df_2b_pivot5)
	fig,ax = plt.subplots()
	sns.heatmap(analysis_df_2b_pivot5,annot=analysis_df_2b_pivot6,vmin=np.min(np.min(analysis_df_2b_pivot5[~inf_mask])),vmax=np.max(np.max(analysis_df_2b_pivot5[~inf_mask])),mask=inf_mask,cmap="YlGnBu",linewidths=.3,ax=ax,cbar_kws={"orientation": "vertical","label":"Mean Time for Broadcast to reach 90% Devices"})
	ax3.set_xlabel("Probability of Transmission to Ordinary Node")
	ax3.set_ylabel("Probability of Transmission to Super Node")
	ax3.annotate('Numbers on Cells indicate - # Devices that could broadcast to 90% or more devices',xy=(0,0),bbox=dict(boxstyle='square',facecolor='white'))
	plt.show()
	################################################################

    ######## Broadcast Algorithm 2 - Analysis 2.c ########
	source_id = 26
	s_list = [1,2,5,10,20,30]
	l = 70
	T_Prob_Super = np.array([1,2,5,10,20,40,50,60,80,90,95,98,99])
	analysis_df_2c = pd.DataFrame(columns=['S','T_Prob_Super','T_Prob_Ordinary','Gini_Load_Super','Gini_Load_Ordinary','Gini_Load_Weak','Mean_Load_Super','Mean_Load_Ordinary','Mean_Load_Weak','Time_90','Chunk Copies','% Devices Reached'])

	### Uncomment only to recalculate entire analysis
	# for idx, s in enumerate(s_list):
	#     for idt, tps in enumerate(T_Prob_Super):
	#         analysis_df_2c.loc[idx*len(s_list)+idt] = [s,tps,100-tps] + list(broadcast_2(devices,source_id,s,l,tps,100-tps))
	#         print(list(analysis_df_2c.loc[idx*len(s_list)+idt]))

	# analysis_df_2c.to_csv('out/Analysis_2c.csv',index=False)

	analysis_df_2c = pd.read_csv('out/Analysis_2c.csv')
	######################################################

    ######### Broadcast Algorithm 2 - Analysis 2.c - Plot 1 ########
	analysis_df_2c_pivot1 = analysis_df_2c.pivot('S','T_Prob_Super','Mean_Load_Super')
	analysis_df_2c_pivot2 = analysis_df_2c.pivot('S','T_Prob_Super','Gini_Load_Super')
	analysis_df_2c_pivot3 = analysis_df_2c.pivot('S','T_Prob_Super','Mean_Load_Ordinary')
	analysis_df_2c_pivot4 = analysis_df_2c.pivot('S','T_Prob_Super','Gini_Load_Ordinary')
	fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(9,12),nrows=2,ncols=2)
	sns.heatmap(analysis_df_2c_pivot1,cmap="YlGnBu",linewidths=.3,ax=ax1,cbar_kws={"orientation": "horizontal","label":"Mean Load on Super Node"})
	ax1.set_xlabel("Probability of Transmission to Ordinary Node")
	ax1.set_ylabel("Super-Node Cutoff % (S)")
	sns.heatmap(analysis_df_2c_pivot2,cmap="YlGnBu",linewidths=.3,ax=ax2,cbar_kws={"orientation": "horizontal","label":"Gini Coeff. for Load on Super Node"})
	ax2.set_xlabel("Probability of Transmission to Ordinary Node")
	ax2.set_ylabel("")
	sns.heatmap(analysis_df_2c_pivot3,cmap="YlGnBu",linewidths=.3,ax=ax3,cbar_kws={"orientation": "horizontal","label":"Mean Load on Ordinary Node"})
	ax3.set_xlabel("Probability of Transmission to Ordinary Node")
	ax3.set_ylabel("Super-Node Cutoff % (S)")
	sns.heatmap(analysis_df_2c_pivot4,cmap="YlGnBu",linewidths=.3,ax=ax4,cbar_kws={"orientation": "horizontal","label":"Gini Coeff. for Load on Ordinary Node"})
	ax4.set_xlabel("Probability of Transmission to Ordinary Node")
	ax4.set_ylabel("")
	plt.show()
	################################################################

	######### Broadcast Algorithm 2 - Analysis 2.c - Plot 2 ########
	analysis_df_2c_pivot5 = analysis_df_2c.pivot('S','T_Prob_Super','Time_90')
	inf_mask = np.isinf(analysis_df_2c_pivot5)
	fig,ax = plt.subplots()
	sns.heatmap(analysis_df_2c_pivot5,vmin=np.min(np.min(analysis_df_2c_pivot5[~inf_mask])),vmax=np.max(np.max(analysis_df_2c_pivot5[~inf_mask])),mask=inf_mask,cmap="YlGnBu",linewidths=.3,ax=ax,cbar_kws={"orientation": "vertical","label":"Time for Broadcast to reach 90% Devices"})
	ax3.set_ylabel("Super-Node Cutoff % (S)")
	ax3.set_xlabel("Probability of Transmission to Super Node ")
	plt.show()
	################################################################

    ######## Broadcast Algorithm 2 - Analysis 2.d ########
	source_id = 26
	s = 0.5
	l_list = [1,5,10,20,40,50,60,80,90,95,99]
	T_Prob_Super = np.array([1,2,5,10,20,40,50,60,80,90,95,98,99])
	analysis_df_2d = pd.DataFrame(columns=['L','T_Prob_Super','T_Prob_Ordinary','Gini_Load_Super','Gini_Load_Ordinary','Gini_Load_Weak','Mean_Load_Super','Mean_Load_Ordinary','Mean_Load_Weak','Time_90','Chunk Copies','% Devices Reached'])

	### Uncomment only to recalculate entire analysis
	# for idx, l in enumerate(l_list):
	#     for idt, tps in enumerate(T_Prob_Super):
	#         analysis_df_2d.loc[idx*len(l_list)+idt] = [l,tps,100-tps] + list(broadcast_2(devices,source_id,s,l,tps,100-tps))
	#         print(list(analysis_df_2d.loc[idx*len(l_list)+idt]))

	# analysis_df_2d.to_csv('out/Analysis_2d.csv',index=False)

	analysis_df_2d = pd.read_csv('out/Analysis_2d.csv')
	######################################################

	######### Broadcast Algorithm 2 - Analysis 2.d - Plot 1 ########
	analysis_df_2d_pivot1 = analysis_df_2d.pivot('L','T_Prob_Super','Mean_Load_Super')
	analysis_df_2d_pivot2 = analysis_df_2d.pivot('L','T_Prob_Super','Gini_Load_Super')
	analysis_df_2d_pivot3 = analysis_df_2d.pivot('L','T_Prob_Super','Mean_Load_Ordinary')
	analysis_df_2d_pivot4 = analysis_df_2d.pivot('L','T_Prob_Super','Gini_Load_Ordinary')
	fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(figsize=(9,12),nrows=2,ncols=2)
	sns.heatmap(analysis_df_2d_pivot1,cmap="YlGnBu",linewidths=.3,ax=ax1,cbar_kws={"orientation": "horizontal","label":"Mean Load on Super Node"})
	ax1.set_xlabel("Probability of Transmission to Ordinary Node")
	ax1.set_ylabel("Weak-Node Cutoff % (L)")
	sns.heatmap(analysis_df_2d_pivot2,cmap="YlGnBu",linewidths=.3,ax=ax2,cbar_kws={"orientation": "horizontal","label":"Gini Coeff. for Load on Super Node"})
	ax2.set_xlabel("Probability of Transmission to Ordinary Node")
	ax2.set_ylabel("")
	sns.heatmap(analysis_df_2d_pivot3,cmap="YlGnBu",linewidths=.3,ax=ax3,cbar_kws={"orientation": "horizontal","label":"Mean Load on Ordinary Node"})
	ax3.set_xlabel("Probability of Transmission to Ordinary Node")
	ax3.set_ylabel("Weak-Node Cutoff % (L)")
	sns.heatmap(analysis_df_2d_pivot4,cmap="YlGnBu",linewidths=.3,ax=ax4,cbar_kws={"orientation": "horizontal","label":"Gini Coeff. for Load on Ordinary Node"})
	ax4.set_xlabel("Probability of Transmission to Ordinary Node")
	ax4.set_ylabel("")
	plt.show()
	################################################################

	######### Broadcast Algorithm 2 - Analysis 2.d - Plot 2 ########
	analysis_df_2d_pivot5 = analysis_df_2d.pivot('L','T_Prob_Super','Time_90')
	inf_mask = np.isinf(analysis_df_2d_pivot5)
	fig,ax = plt.subplots()
	sns.heatmap(analysis_df_2d_pivot5,vmin=np.min(np.min(analysis_df_2d_pivot5[~inf_mask])),vmax=np.max(np.max(analysis_df_2d_pivot5[~inf_mask])),mask=inf_mask,cmap="YlGnBu",linewidths=.3,ax=ax,cbar_kws={"orientation": "vertical","label":"Time for Broadcast to reach 90% Devices"})
	ax3.set_ylabel("Weak-Node Cutoff % (L)")
	ax3.set_xlabel("Probability of Transmission to Super Node ")
	plt.show()
	################################################################

if __name__ == '__main__':
    if len(sys.argv) < 6:
        broadcast_2_analysis()
    else:
        print(broadcast_2(devices,float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])))