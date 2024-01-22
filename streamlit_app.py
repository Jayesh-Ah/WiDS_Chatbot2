{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import streamlit as st\nimport spacy as sp\nimport fitz\nimport numpy as np\nfrom sklearn.metrics.pairwise import cosine_similarity\nnlp = spacy.load(\"en_core_web_md\")\n\ndef main():\n\n  #STREAMLIT APP\n  st.header('''VartaVault Maestro''')\n  st.sidebar.header(\"PDF Chatbot\")\n  st.sidebar.markdown('''\n  ## About\n  Elevate your document experience with this sleek PDF Chatbot, powered by Python, Spacy, Langchain, and Streamlit. Unleash the synergy of NLP and intuitive design for a seamless interaction journey.''')\n  pdf = st.sidebar.file_uploader(\"Upload or drag PDF here\", type = \"pdf\", accept_multiple_files=True)\n  st.sidebar.button(\"Upload\")\n\n  #EXTRACTING TEXT\n  def extract(doc):\n    file = fitz.open(doc)\n    words = \"\"\n    for page_num in range(file.page_count):\n      page = doc[page_num]\n      words += page.get_text()\n    doc.close()\n    return words\n  \n\nif __name__ == '__main__':\n  main()\n  ","metadata":{"execution":{"iopub.status.busy":"2024-01-22T13:42:05.618251Z","iopub.execute_input":"2024-01-22T13:42:05.618982Z","iopub.status.idle":"2024-01-22T13:42:06.050392Z","shell.execute_reply.started":"2024-01-22T13:42:05.618947Z","shell.execute_reply":"2024-01-22T13:42:06.048740Z"},"trusted":true},"execution_count":1,"outputs":[{"traceback":["\u001b[0;31m---------------------------------------------------------------------------\u001b[0m","\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)","Cell \u001b[0;32mIn[1], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mstreamlit\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mst\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mspacy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01msp\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mfitz\u001b[39;00m\n","\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'"],"ename":"ModuleNotFoundError","evalue":"No module named 'streamlit'","output_type":"error"}]}]}