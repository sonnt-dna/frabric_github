#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark.sql import HiveContext
from pyspark.streaming import StreamingContext

sc = None
spark = None
sqlContext = None

def init_spark():
    import os
    from py4j.java_gateway import java_import, JavaGateway, GatewayClient
    from pyspark.conf import SparkConf
    from pyspark.context import SparkContext
    from pyspark.sql import SparkSession, SQLContext
    global sc, spark, sqlContext
    gateway_port = int(os.environ["PYSPARK_GATEWAY_PORT"])

    try:
        from py4j.java_gateway import GatewayParameters
        gateway_secret = os.environ["PYSPARK_GATEWAY_SECRET"]
        gateway = JavaGateway(gateway_parameters=GatewayParameters(
            port = gateway_port, auth_token=gateway_secret, auto_convert=True))
    except:
        gateway = JavaGateway(GatewayClient(port=gateway_port), auto_convert=True)

    java_import(gateway.jvm, "org.apache.spark.SparkConf")
    java_import(gateway.jvm, "org.apache.spark.api.java.*")
    java_import(gateway.jvm, "org.apache.spark.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.mllib.api.python.*")
    java_import(gateway.jvm, "org.apache.spark.sql.*")
    java_import(gateway.jvm, "org.apache.spark.sql.hive.*")
    java_import(gateway.jvm, "org.apache.spark.sql.api.python.*")
    java_import(gateway.jvm, "scala.Tuple2")

    jsc = gateway.entry_point.sc()
    jconf = gateway.entry_point.sc().getConf()
    jsqlc = gateway.entry_point.hivectx() if gateway.entry_point.hivectx() is not None \
        else gateway.entry_point.sqlctx()

    conf = SparkConf(_jvm=gateway.jvm, _jconf=jconf)
    sc = SparkContext(jsc=jsc, gateway=gateway, conf=conf)
    spark = SparkSession(sc, gateway.jvm.org.apache.livy.repl.Session.getSparkSession(os.environ["LIVY_SPARK_SESSION_SOURCE_ID"]))
    sqlContext = SQLContext(sc, spark, jsqlc)

init_spark()
del init_spark


# In[ ]:


from notebookutils import prepare
prepare(global_namespace=globals())


# In[ ]:





# In[ ]:


("2", """Job group for statement 2:
import notebookutils

# Personalize Session
from notebookutils.common.initializer import initializeLHContext
initializeLHContext()

# see also: https://msdata.visualstudio.com/A365/_git/NotebookUtils?path=/python/notebookutils/__init__.py&_a=contents&version=GBmaster
notebookutils.prepare(globals())
""")



# In[ ]:


import notebookutils

# Personalize Session
from notebookutils.common.initializer import initializeLHContext
initializeLHContext()

# see also: https://msdata.visualstudio.com/A365/_git/NotebookUtils?path=/python/notebookutils/__init__.py&_a=contents&version=GBmaster
notebookutils.prepare(globals())


# In[ ]:





# In[ ]:


("3", """Job group for statement 3:
import sys
sys.path.append('/lakehouse/default/Files')
sys.path.append('../')

import nbformat as nbf
import os

#Put the name of file
output_filename =  "Testing28"

# Thay đổi đường dẫn và tên file output
output_dir = '/lakehouse/default/Files'
output_filename_notebook = f'{output_filename}.ipynb'
output_path = os.path.join(output_dir, output_filename_notebook)

# Kiểm tra sự tồn tại của file
if not os.path.exists(output_path):
    # Tạo một notebook mới
    new_notebook = nbf.v4.new_notebook()

    # Lấy danh sách các cell trong notebook đang mở
    cells = get_ipython().history_manager.get_range(session=0)

    # Thêm từng cell vào notebook mới
    for _, _, cell in cells:
        new_notebook.cells.append(nbf.v4.new_code_cell(cell))

    # Lưu notebook mới vào file .ipynb
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(nbf.writes(new_notebook))

    print(f'File {output_filename_notebook} đã được tạo thành công.')
else:
    print(f'File {output_filename_notebook} đã tồn tại.')

##...""")



# In[ ]:


import sys
sys.path.append('/lakehouse/default/Files')
sys.path.append('../')

import nbformat as nbf
import os

#Put the name of file
output_filename =  "Testing28"

# Thay đổi đường dẫn và tên file output
output_dir = '/lakehouse/default/Files'
output_filename_notebook = f'{output_filename}.ipynb'
output_path = os.path.join(output_dir, output_filename_notebook)

# Kiểm tra sự tồn tại của file
if not os.path.exists(output_path):
    # Tạo một notebook mới
    new_notebook = nbf.v4.new_notebook()

    # Lấy danh sách các cell trong notebook đang mở
    cells = get_ipython().history_manager.get_range(session=0)

    # Thêm từng cell vào notebook mới
    for _, _, cell in cells:
        new_notebook.cells.append(nbf.v4.new_code_cell(cell))

    # Lưu notebook mới vào file .ipynb
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(nbf.writes(new_notebook))

    print(f'File {output_filename_notebook} đã được tạo thành công.')
else:
    print(f'File {output_filename_notebook} đã tồn tại.')

### Tạo file python từ file notebook đã tạo

import os
import nbformat
from nbconvert import PythonExporter

filepath = f"{output_dir}/{output_filename_notebook}"
output_filename_py = f'{output_filename}.py'

# Tạo đối tượng PythonExporter
exporter = PythonExporter()

# Đọc nội dung file notebook và chuyển đổi nó thành mã Python
with open(filepath, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)
    body, _ = exporter.from_notebook_node(nb)

# Loại bỏ các dòng không mong muốn
unwanted_lines = [
    '',
    '',
    '',
    '("16", """Job group for statement 16'
]
for line in unwanted_lines:
    body = body.replace(line, '')

# Kiểm tra sự tồn tại của file Python
python_code_path = os.path.join(output_dir, output_filename_py)
if os.path.exists(python_code_path):
    print('File Python đã tồn tại.')
else:
    # Tạo mới file Python và ghi nội dung vào
    try:
        with open(python_code_path, 'w', encoding='utf-8') as f:
            f.write(body)
        print('File Python đã được tạo thành công.')
    except Exception as e:
        print(f'Lỗi khi tạo file Python: {str(e)}')

# Kiểm tra lại sự tồn tại của file Python
if os.path.exists(python_code_path):
    print('File Python đã tồn tại sau khi tạo.')
else:
    print('File Python vẫn không tồn tại.')

