# Q3_Data

## <u>Number of Hops Table</u>

|         | ETHZ | Univ_of_Waterloo | Univ_of_Cape_Town | IIT_Delhi | Google | Facebook |
|:-------:|:----:|:----------------:|:-----------------:|:---------:|:------:|:--------:|
| Canada  | 30+  |       30+        |        30+        |    30+    |   13   |   30+    |
| Sweden  | 30+  |       30+        |        30+        |    30+    |   14   |   30+    |
|  UK     | 30+  |       30+        |        30+        |    30+    |   11   |   30+    |
|  NZ     | 30+  |       30+        |        30+        |    30+    |   17   |   30+    |
|  SA     | 30+  |       30+        |        30+        |    30+    |   12   |   30+    |
| Germany | 30+  |       30+        |        30+        |    30+    |   14   |   30+    |

---

## <u>Latency Table</u>

**Note**: Latency reported in 'ms'

|         | ETHZ | Univ_of_Waterloo | Univ_of_Cape_Town | IIT_Delhi | Google | Facebook |
|:-------:|:----:|:----------------:|:-----------------:|:---------:|:------:|:--------:|
| Canada  |  -   |        -         |         -         |     -     | 69.168 |    -     |
| Sweden  |  -   |        -         |         -         |     -     | 124.15 |    -     |
|  UK     |  -   |        -         |         -         |     -     | 82.294 |    -     |
|  NZ     |  -   |        -         |         -         |     -     | 223.47 |    -     |
|  SA     |  -   |        -         |         -         |     -     | 238.28 |    -     |
| Germany |  -   |        -         |         -         |     -     | 87.848 |    -     |

---

## <u>Hops - Latency Correlation</u>

There is not much of data to deduce correlations perfectly. But still it can be observed that as the number of hops increase, latency increases (except in 2 cases).

---

## <u>Latency Table using Cellular Data</u>

**Note**: Latency reported in 'ms'  

|    Destination    | Total_Hops | Hops_inside_local_ISP | Total_Latency | Latency_inside_local_ISP |
|:-----------------:|:----------:|:---------------------:|:-------------:|:------------------------:|
| ETHZ              |    30+     |          10           |       -       |         81.901           |
| Univ of Waterloo  |    30+     |          10           |       -       |         116.176          |
| Univ of Cape Town |    30+     |          10           |       -       |         89.552           |
| IITD              |    30+     |          8            |       -       |         68.053           |
| Google            |    30+     |          30+          |       -       |           -              |
| Facebook          |    30+     |          10           |       -       |         85.075           |

---