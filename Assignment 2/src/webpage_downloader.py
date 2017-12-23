from HAR_parser import *
import socket, mimetypes, os, time
import pandas as pd
from threading import Thread, current_thread, active_count

################# Helper Methods ################
def make_request_str(request_dict):
    method = request_dict['method']
    if method != "GET":
        print("Non-GET method - ",method)
        return 'Invalid'
    url = request_dict['url']
    http_ver = request_dict['httpVersion']
    # header_str = ''.join([x['name'] + ": " + x['value'] + "\r\n" for x in request_dict['headers'] if x['name'].lower() != 'accept-encoding'])
    header_str = ''.join([x['name'] + ": " + x['value'] + "\r\n" for x in request_dict['headers']])
    request_str = method + " " + url + " " + http_ver + "\r\n" + header_str + "\r\n"
    return bytes(request_str,'utf-8')

def parse_response_header(response_header):
    response_split = response_header.split('\r\n')
    response_dict = {}
    for x in response_split:
        y = x.split(': ')
        if len(y) > 1:
            response_dict.update({str(y[0]):str(y[1])})
    return response_dict

def get_file_extension(mime_type):
    ext = mimetypes.guess_extension(mime_type)
    if ext == None:
        mime_split = mime_type.split('/')
        ext = '.' + (mime_split[-1] if mime_split[0] == 'image' or mime_split[0] == 'audio' or mime_split[0] == 'video' else 'txt')
    else:
        if 'javascript' in ext: ext = ".js"
    if ext == '.jpe': ext = '.jpg'
    if ext == '.htm': ext = '.html'
    return ext
#################################################

############# Object Download Method ############
def get_object(sck,request_dict,domain,reconnect):
    request = make_request_str(request_dict)
    if request != "Invalid":
        for i in range(3):
            try:
                if reconnect == 0:
                    sck.connect((domain,80))
                    # print("Connected to ",domain," on port 80")
                    break
            except OSError as ex:
                if '106' in str(ex):
                   break                # Already Connected
                else:
                    print("Error occured - ",str(ex),": Trying to reconnect ",str(i+1),"/3")
                    if i == 2: return ("Error occured","")
        try:
            sck.send(request)
            response = sck.recv(1024)
            header_end = response.find(bytes("\r\n\r\n",'utf-8'))+len("\r\n\r\n")
            response_header = response[:header_end] if header_end > 3 else response
            content_start = 1 if header_end > 3 else -1
            content = response[header_end:] if content_start == 1 else ""
            while len(response) > 0:
                response = sck.recv(1024)
                if content_start == -1:
                    add_response = response_header + response
                    header_end = add_response.find(bytes("\r\n\r\n",'utf-8'))+len("\r\n\r\n")
                    response_header = add_response[:header_end] if header_end > 3 else add_response
                    content_start = 1 if header_end > 3 else -1
                    content = add_response[header_end:] if content_start == 1 else ""
                else:
                    content = content + response
            return (str(response_header,'utf-8'),content)
        except:
            return ("Error occured","")
    else:
        return ("Invalid request encountered","")
#################################################

############ Webpage Download Method ############
def socket_thread(domain,request_dicts,out_folder,max_obj):
    try:
        os.makedirs(out_folder)
    except:
        print("Unable to create directory")
        return
    # print("Thread started: ",current_thread().name,"\nNumber of objects to download = ",len(request_dicts))
    sck = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    for idx, req_dict in enumerate(request_dicts):
        if idx % max_obj == 0:
            sck.close()
            sck = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        start_time = time.time()
        response = get_object(sck,req_dict,domain,(idx % max_obj))
        end_time = time.time()
        if response[0][:5] != "Error" and response[0][:7] != "Invalid":
            resp_dict = parse_response_header(response[0])
            resp_dict.update({'Response Time': (end_time - start_time)*1000})
            with open(out_folder + str(idx) + "_header.txt","w") as f:
                resp_header_str = "\n".join([str(key) + ": " + str(resp_dict[key]) for key in resp_dict.keys()])
                f.write(resp_header_str)
            content_type = [val for key, val in resp_dict.items() if key.lower() == 'content-type']
            mime_type = content_type[0].split(";")[0] if len(content_type) > 0 else -1
            ext = get_file_extension(mime_type) if mime_type != -1 else '.txt'
            if type(response[1]) == bytes:
                with open(out_folder + str(idx) + ext,"wb") as f:
                    f.write(response[1])
            elif type(response[1]) == str:
                with open(out_folder + str(idx) + ext,"w") as f:
                    f.write(response[1])
    sck.close()
    # print("Thread ended: ",current_thread().name)
    if (active_count() % 10 == 0): print("Active connections = ",str(active_count() - 1))
    return

def get_webpage(json_filepath,max_tcp,max_obj):
    data, essential_df, complete_df = har_parser(json_filepath)
    out_folder = 'out/Webpages' + json_filepath.split('/')[2] + ' - ' + json_filepath.split('/')[3].split('.')[0] + '/'
    deleteDir(out_folder)
    domain_grouping = complete_df.groupby('domain')
    all_threads = []
    start_time = time.time()
    print("Starting Download")
    for idx, x in enumerate(domain_grouping):
        out_folder_domain = out_folder + x[0] + "/"
        sorted_data = x[1].sort_values(by='contentSize',axis=0,ascending=True)
        scks_count = min(max_tcp,len(sorted_data))
        req_chunks_idx = [list(range(len(x[1])))[i::scks_count] for i in range(scks_count)]
        for idk in range(scks_count):
            request_dicts = [sorted_data.loc[sorted_data.index[idk]]['request_dict'] for i in req_chunks_idx[idk]]
            t = Thread(name=x[0]+" - Socket_" + str(idk),target=socket_thread,args=(x[0],request_dicts,out_folder_domain + "Socket_" + str(idk) + "/",max_obj))
            all_threads.append(t)
            t.start()
    print("Started all",str(len(all_threads)),"connections in",(time.time()-start_time),"s")
    for t in all_threads:
        t.join()
    end_time = time.time()
    print("Time to download webpage = ",str(end_time - start_time))
#################################################
        
if __name__ == '__main__':
    json_filepath = 'data/HAR Dumps/' + ('Indian Express/unthrottled_windows' if len(sys.argv) == 1 else ' '.join(sys.argv[1:])) + '.har'
    max_tcp = 5
    max_obj = 5
    get_webpage(json_filepath,max_tcp,max_obj)