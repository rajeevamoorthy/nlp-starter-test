import tarfile
import base64
import os

try:
    working_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(working_dir)

    filename = 'db.sqlite.base64.tar.gz'

    comp_file = tarfile.open(parent_dir + '/problem_statement/' + filename, mode='r:gz')

    for item in comp_file.getmembers():
        if item.isfile():
            db64_file = comp_file.extractfile(item)
            with open(working_dir + '/db.sqlite', 'wb') as db2:
                db2.write(base64.b64decode(db64_file.read()))

except Exception as e:
    print(e)
