## Create and activate a virtual env:

    virtualenv -p python3 env-assignment
    . env*/bin/activate

## Install requirements

    pip install -r requirements.txt

## Execute script to convert sqlite db from base64 to binary

    python sqlite_decode_script.py
