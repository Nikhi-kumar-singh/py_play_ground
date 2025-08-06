from nbclient import NotebookClient
from nbformat import read, write

ipynb_file="t001.ipynb"

with open(ipynb_file,encoding='utf-8') as f:
    nb = read(f, as_version=4)

client = NotebookClient(nb)
client.execute()

with open("executed_notebook.ipynb", "w",encoding='utf-8') as f:
    write(nb, f)
