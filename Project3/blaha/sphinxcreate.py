'''
In Project3 type:

Fannys-MacBook-Pro:Project3 fannyandersson$ sphinx-apidoc . --full -o docs -H 'Project3' -A 'Project3' -V '1.0'


Open config.py in docs directory and change from:

# import os
# import sys
# sys.path.insert(0, '/Users/fannyandersson/PycharmProjects/Python-course/Project3')

To:

import os
import sys
# sys.path.insert(0, '/Users/fannyandersson/PycharmProjects/Python-course/Project3')
sys.path.insert(0, os.path.abspath('../'))


Then type

Fannys-MacBook-Pro:Project3 fannyandersson$ sphinx-build -b html docs docs/_build


If something has been changed and needs to be updated in htlm, run:

Fannys-MacBook-Pro:Project3 fannyandersson$ sphinx-build -b html docs docs/_build

again.


To open html in browser open index.html in docs/_build and click on preferred browser icon.

'''