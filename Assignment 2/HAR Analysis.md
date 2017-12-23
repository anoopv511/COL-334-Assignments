# HAR Analysis

## Question 3.b.iii.5

Maximum Goodput is more than the Average Goodput in almost all cases. But we observe that in most cases Maximum Goodput is skewed by data that has close to 0 receive time.

## Question 3.b.iii.7

In most cases Average Network Goodput is less than Maximum Network Goodput. This means that our network is not being utilized fully. This might be due to some slow servers pulling down the average.Also cases where receive time is close to 0 skew the Maximum Goodput value.

## Question 3.b.iv.1

In all cases mobile browser opens less TCP conections than desktop browser because of lesser computational power of mobile as compared to desktop. 

## Question 3.b.iv.2

In most cases Slow 3G opens more TCP connections per Domain than Fast 3G so as to parallelize more downloads for better download speeds.

## Question 3.b.iv.3

In most cases we observed that desktop browser imposed a greater cap on number of TCP connections opened as compared to mobile browser due to computational reasons.

## Question 3.c

- First Estimate:  
    This is less than page load time observed from har file in all cases.

- Second Estimate:  
    This is estimate is varying as compared to page load time observed from har file. This is due to skewed measurement of the Maximum Goodput.

- Deviation from observed page load time:  
    While calculating these estimates, we are making assumptions that parallel TCP connections can be opened in any amount. But in reality this is not possible because we don't know before hand how many TCP connections will be required.

## Question 4.c

We observed that page load time from har file is lower than when we used sockets based downloader script. This is because we did not use compressed encoding like 'gzip'. We also observed that the downloader script is performing slightly better with higher max_TCP but performing similarly with higher max_obj. This is due to relatively slower web servers as they are not able to push enough data (Average Network Goodput < Max. Network Goodput).

## Question 4.d

Number of objects that need to be downloaded is an important parameter in deciding download speed. By knowing this is advance we can open multiple TCP connections effectively.