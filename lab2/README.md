## Лабораторная работа (весна 2022 г.)
<b>Дано:</b>
<br>
CSV-файл с числовыми значениями ЧСС, калорийности, MET по нескольких людям за промежуток времени (например,
[all_data_merged_interpolated_12_users_no_cluster_1.csv](https://h1bymask.github.io/2022/all_data_merged_interpolated_12_users_no_cluster.7z)
), предварительно размещенный на HDFS.
<br>
<br>
<b>Требуется:</b>
<br>
Провести кластеризацию данных. После этого вычислить моду и медиану ЧСС и MET для каждого кластера и для всего набора данных.
<br>
Предложения:
<br>
Ипользовать известные вам методы Pandas
<br>
Метод k средних
<br>
Использовать hadoop-streaming jar, либо модуль Python hdfs, либо Spark (либо его Python-биндинг, либо Spark SQL).
