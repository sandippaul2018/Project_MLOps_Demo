#!/usr/bin/env python
import subprocess
import sys

# Run streamlit with environment variable to skip analytics
import os
os.environ['STREAMLIT_TELEMETRY_OPTOUT'] = 'true'

# Run the streamlit command
subprocess.run([
    sys.executable, '-m', 'streamlit', 'run', 'src/app.py',
    '--client.toolbarMode=minimal'
])
