import json, sys, os
import pandas as pd
import numpy as np

################# Helper Methods ################
def get_domain(request):
    domain = [x['value'] for x in request['headers'] if x['name'] == 'Host']
    if len(domain) == 1:
        return domain[0]
    url = request['url']
    if '/' in url:
        url_split = url.split('/')
        if 'http' in url_split[0]:
            return url_split[2]
        else:
            return url_split[0]
    else:
        return url

def deleteDir(dirPath):
    deleteFiles = []
    deleteDirs = []
    for root, dirs, files in os.walk(dirPath):
        for f in files:
            deleteFiles.append(os.path.join(root, f))
        for d in dirs:
            deleteDirs.append(os.path.join(root, d))
    for f in deleteFiles:
        try:
            os.remove(f)
        except:
            pass
    for d in deleteDirs:
        try:
            deleteDir(d)
        except:
            pass
    try:
        os.rmdir(dirPath)
    except:
        pass
#################################################

################ HAR Data Parsing ###############
def har_parser(json_filepath):
    with open(json_filepath) as json_data:
        data = json.load(json_data)

    essential_df_col = ['connection_id','domain','total_time','requestSize','responseStatus','contentSize','request_dict']
    essential_df = pd.DataFrame(columns=essential_df_col)
    timing_df = pd.DataFrame()

    for idx, entry in enumerate(data['log']['entries']):
        t_df = pd.DataFrame(entry['timings'],index=[0])
        connection_id = entry.get('connection',np.nan)
        total_time = sum(t_df[t_df >= 0].iloc[0].fillna(0))
        domain = get_domain(entry['request'])
        requestSize = entry['request']['headersSize'] + entry['request']['bodySize']
        responseStatus = entry['response']['status']
        contentSize = entry['response']['content']['size']
        request_dict = entry['request']
        essential_df.loc[idx] = [connection_id,domain,total_time,requestSize,responseStatus,contentSize,request_dict]
        timing_df = timing_df.append(t_df, ignore_index=True)

    essential_df = essential_df.reindex_axis(essential_df_col + list(essential_df.columns.drop(essential_df_col)), axis=1)
    complete_df = pd.concat([essential_df,timing_df], axis=1)
    return data, essential_df, complete_df
#################################################

################ Domain Analysis ################
def domain_wise_analyser(data,complete_df):
    domain_grouping = complete_df.groupby('domain')
    analysis_df_col =['Domain','# Objects','# Non-Zero Objects','Content Size','# Connections',"Connection Analysis"]
    analysis_df = pd.DataFrame(columns=analysis_df_col)
    for idx, x in enumerate(domain_grouping):
        domain_name = x[0]
        content_size = x[1]['contentSize'].sum()
        object_count = len(x[1])
        non_zero_object_count = (x[1]['contentSize'] > 0).sum()
        connection_count = len(x[1])
        connection_grouping = x[1].groupby('connection_id')
        connection_analysis_df = pd.DataFrame(columns=['Connection ID','# Objects','# Non-Zero Objects','Content Size','DNS'])
        for idy, y in enumerate(connection_grouping):
            connection_analysis_df.loc[idy] = [y[0],len(y[1]),(y[1]['contentSize'] > 0).sum(),y[1]['contentSize'].sum(),list(y[1]['dns'])]
        analysis_df.loc[idx] = [domain_name,object_count,non_zero_object_count,content_size,connection_count,connection_analysis_df]
    return analysis_df

def print_domain_wise_analysis(analysis_df):
    analysis_df_col = analysis_df.columns
    print_domain_analysis = input("\33[93mEnter 'y' to see Domain-wise analysis: \33[0m")
    if print_domain_analysis == 'y':
        print('-------------------------------------------------------')
        for x in range(analysis_df.shape[0]):
            print('\33[91m',analysis_df_col[0],'\33[0m',"\t\t-\t",'\33[95m',analysis_df.iloc[x][0],'\33[0m')
            print('\33[92m',analysis_df_col[1],'\33[0m',"\t\t-\t",'\33[94m',analysis_df.iloc[x][1],'\33[0m')
            print('\33[92m',analysis_df_col[2],'\33[0m',"\t-\t",'\33[94m',analysis_df.iloc[x][2],'\33[0m')
            print('\33[92m',analysis_df_col[3],'\33[0m',"\t\t-\t",'\33[94m',analysis_df.iloc[x][3],'\33[0m')
            print('\33[92m',analysis_df_col[4],'\33[0m',"\t-\t",'\33[94m',analysis_df.iloc[x][4],'\33[0m')
            print('\33[93m',analysis_df_col[5],'\33[0m')
            print(analysis_df.iloc[x][5].to_string(index=False))
            print('-------------------------------------------------------')

    print('\33[92mTotal Number of Domains\33[0m - \33[95m',analysis_df.shape[0],'\33[0m')
    print('\33[92mTotal Number of Objects Downloaded\33[0m - \33[95m',analysis_df['# Objects'].sum(),'\33[0m')
    print('\33[92mTotal Number of Non-Zero Objects Downloaded\33[0m - \33[95m',analysis_df['# Non-Zero Objects'].sum(),'\33[0m')
    print('\33[92mTotal Content Downloaded\33[0m - \33[95m{0:.2f} MB\33[0m'.format(analysis_df['Content Size'].sum()/1024**2))
    print('\33[92mPage Load Time\33[0m - \33[95m{0:.3f} s\33[0m'.format(data['log']['pages'][0]['pageTimings']['onLoad']/1000))
#################################################

############ TCP Connection Analysis ############
def connection_wise_analyser(data,complete_df):
    connection_grouping = complete_df.groupby('connection_id')
    tcp_analysis_df_col = ['Connection ID','Connect Time','Average Wait Time','Total Receive Time','Total Content Size','Average Goodput','Maximum Goodput']
    tcp_analysis_df = pd.DataFrame(columns=tcp_analysis_df_col)
    for idx, x in enumerate(connection_grouping):
        connect_time = np.max(x[1]['connect'])
        avg_wait_time = x[1]['wait'].sum()/len(x[1])
        total_receive_time = x[1]['receive'].sum()
        total_content_size = x[1]['contentSize'].sum()
        avg_goodput = 0 if total_content_size == 0 else float('inf') if total_receive_time == 0 else total_content_size/total_receive_time
        max_object_size_idx = np.argmax(x[1]['contentSize'])
        max_goodput = 0 if x[1]['contentSize'][max_object_size_idx] == 0 else float('inf') if x[1]['receive'][max_object_size_idx] == 0 else x[1]['contentSize'][max_object_size_idx]/x[1]['receive'][max_object_size_idx]
        tcp_analysis_df.loc[idx] = [x[0],connect_time,avg_wait_time,total_receive_time,total_content_size,avg_goodput,max_goodput]
    return tcp_analysis_df

def print_connection_wise_analysis(tcp_analysis_df):
    tcp_analysis_df_col = tcp_analysis_df.columns
    print_tcp_analysis = input("\33[93mEnter 'y' to see TCP Connection-wise analysis: \33[0m")
    if print_tcp_analysis == 'y':
        print('-------------------------------------------------------')
        for x in range(tcp_analysis_df.shape[0]):
            print('\33[91m',tcp_analysis_df_col[0],'\33[0m',"\t-\t",'\33[95m',tcp_analysis_df.iloc[x][0],'\33[0m')
            print('\33[92m',tcp_analysis_df_col[1],'\33[0m',"\t\t-\t",'\33[94m',tcp_analysis_df.iloc[x][1],'\33[0m')
            print('\33[92m',tcp_analysis_df_col[2],'\33[0m',"\t-\t",'\33[94m',tcp_analysis_df.iloc[x][2],'\33[0m')
            print('\33[92m',tcp_analysis_df_col[3],'\33[0m',"\t-\t",'\33[94m',tcp_analysis_df.iloc[x][3],'\33[0m')
            print('\33[92m',tcp_analysis_df_col[4],'\33[0m',"\t-\t",'\33[94m',tcp_analysis_df.iloc[x][4],'\33[0m')
            print('\33[92m',tcp_analysis_df_col[5],'\33[0m',"\t-\t",'\33[94m',tcp_analysis_df.iloc[x][5],'\33[0m')
            print('\33[92m',tcp_analysis_df_col[6],'\33[0m',"\t-\t",'\33[94m',tcp_analysis_df.iloc[x][6],'\33[0m')
            print('-------------------------------------------------------')

    total_network_receive_time = data['log']['pages'][0]['pageTimings']['onLoad']/1000
    total_network_content_size = tcp_analysis_df['Total Content Size'].sum()/1024**2
    print('\33[92mTotal Number of Connections\33[0m - \33[95m',tcp_analysis_df.shape[0],'\33[0m')
    print('\33[92mTotal Content Downloaded\33[0m - \33[95m{0:.2f} MB\33[0m'.format(total_network_content_size))
    print('\33[92mTotal Receive Time\33[0m - \33[95m{0:.2f} s\33[0m'.format(total_network_receive_time))
    print('\33[92mAverage Network Goodput\33[0m - \33[95m{0:.2f} MB/s\33[0m'.format(total_network_content_size/total_network_receive_time))
    print('\33[92mMaximum Network Goodput\33[0m - \33[95m{0:.2f} MB/s\33[0m'.format(np.max(tcp_analysis_df['Maximum Goodput'][tcp_analysis_df['Maximum Goodput'] < float('inf')]*(1000/(1024**2)))))
#################################################

if __name__ == '__main__':
    json_filepath = 'data/HAR Dumps/' + ('Indian Express/unthrottled_windows' if len(sys.argv) == 1 else ' '.join(sys.argv[1:])) + '.har'
    data, essential_df, complete_df = har_parser(json_filepath)
    analysis_df = domain_wise_analyser(data,complete_df)
    tcp_analysis_df = connection_wise_analyser(data,complete_df)
    print_domain_wise_analysis(analysis_df)
    print_connection_wise_analysis(tcp_analysis_df)

    connection_grouping = complete_df.groupby('connection_id')
    time_per_connection_1 = []
    for idx, x in enumerate(connection_grouping):
        max_dns = np.max(x[1]['dns'])
        max_wait = np.max(x[1]['wait'])
        connect_time = np.max(x[1]['connect'])
        total_send = x[1]['send'].sum()
        total_recv = x[1]['receive'].sum()
        time_per_connection_1.append(max_dns + max_wait + connect_time + total_send + total_recv)
    alternate_page_load_time_1 = np.max(time_per_connection_1)
    print('\33[92mAlternate Page Load Time (Q3.c.i) - \33[95m{0:.3f} s'.format(alternate_page_load_time_1/1000),"\33[0m")

    domain_grouping = complete_df.groupby('domain')
    time_per_connection_2 = []
    for idx, x in enumerate(domain_grouping):
        total_content_size = x[1]['contentSize'].sum()
        max_object_size_idx = np.argmax(x[1]['contentSize'])
        max_goodput = 0 if x[1]['contentSize'][max_object_size_idx] == 0 else float('inf') if x[1]['receive'][max_object_size_idx] == 0 else x[1]['contentSize'][max_object_size_idx]/x[1]['receive'][max_object_size_idx]
        total_recv = 0 if total_content_size == 0 else -1 if max_goodput == 0 else total_content_size/max_goodput
        time_per_connection_2.append(total_recv)
    alternate_page_load_time_2 = np.max(time_per_connection_2)
    print('\33[92mAlternate Page Load Time (Q3.c.iii) - \33[95m{0:.3f} s'.format(alternate_page_load_time_2/1000),"\33[0m")

    out_folder = 'out/HAR Analysis/' + json_filepath.split('/')[2] + ' - ' + json_filepath.split('/')[3].split('.')[0] + '/'
    deleteDir(out_folder)
    os.makedirs(out_folder)
    analysis_df.to_csv(out_folder + 'domain_analysis.csv')
    tcp_analysis_df.to_csv(out_folder + 'tcp_analysis.csv')
    with open(out_folder + 'analysis.txt',"w") as f:
        f.write('Total Number of Domains - ' + str(analysis_df.shape[0]) + "\n")
        f.write('Total Number of Objects Downloaded - ' + str(analysis_df['# Objects'].sum()) + "\n")
        f.write('Total Number of Non-Zero Objects Downloaded - ' + str(analysis_df['# Non-Zero Objects'].sum()) + "\n")
        f.write('Total Content Downloaded - {0:.2f} MB'.format(analysis_df['Content Size'].sum()/1024**2) + "\n")
        f.write('Page Load Time - {0:.3f} s'.format(data['log']['pages'][0]['pageTimings']['onLoad']/1000) + "\n")
        total_network_receive_time = data['log']['pages'][0]['pageTimings']['onLoad']/1000
        total_network_content_size = tcp_analysis_df['Total Content Size'].sum()/1024**2
        f.write('Total Number of Connections - ' + str(tcp_analysis_df.shape[0]) + "\n")
        f.write('Total Receive Time - {0:.2f} s'.format(total_network_receive_time) + "\n")
        f.write('Average Network Goodput - {0:.2f} MB/s'.format(total_network_content_size/total_network_receive_time) + "\n")
        f.write('Maximum Network Goodput - {0:.2f} MB/s'.format(np.max(tcp_analysis_df['Maximum Goodput'][tcp_analysis_df['Maximum Goodput'] < float('inf')]*(1000/(1024**2)))) + "\n")
        f.write('Alternate Page Load Time (Q3.c.i) - {0:.3f} s'.format(alternate_page_load_time_1/1000) + "\n")
        f.write('Alternate Page Load Time (Q3.c.iii) - {0:.3f} s'.format(alternate_page_load_time_2/1000) + "\n")